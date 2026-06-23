"""
g1_walk_env.py  — MuJoCo walking environment for Unitree G1 (29 DoF).

Translated from Isaac Lab  velocity_env_cfg.py  (unitree_rl_lab)
to Gymnasium MujocoEnv + Stable-Baselines3 PPO.

Implements:
  • Velocity-tracking rewards  (body-frame, relative to platform surface)
  • Phase-based gait rewards   (alternating foot contacts, 0.8 s period)
  • Foot clearance & slide penalties
  • Joint-deviation penalties  (arms, waist, hip-roll/yaw)
  • Observation history stack  (5 frames x 96 = 480-dim obs)
  • Curriculum learning        (velocity-command range + platform perturbation)
  • Domain randomisation       (friction, velocity pushes)
  • Platform centering         (keeps robot on 2m x 2m platform)

Usage:
    from envs.g1_walk_env import G1WalkEnv, make_walk_env
    env = G1WalkEnv(render_mode="human")
"""

from collections import deque
import mujoco
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from envs.g1_walk_config import *


# ═══════════════════════════════════════════════════════════════════════
class G1WalkEnv(MujocoEnv):
    """Velocity-tracking locomotion for the G1 on a sliding platform."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    # ───────────────────── init ─────────────────────
    def __init__(self, render_mode=None):
        observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,),           # 480
            dtype=np.float64,
        )
        super().__init__(
            model_path=MODEL_PATH,
            frame_skip=5,               # env dt = 0.002 × 5 = 0.01 s
            observation_space=observation_space,
            render_mode=render_mode,
        )
        # Override action space to 29 joints (exclude 2 platform actuators)
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(29,), dtype=np.float32,
        )

        # ── pre-compute IDs ──
        self._init_body_and_geom_ids()

        # ── joint limits  (for dof_pos_limits penalty) ──
        self._joint_limits_low  = np.zeros(29, dtype=np.float64)
        self._joint_limits_high = np.zeros(29, dtype=np.float64)
        for i in range(29):
            jnt_id = i + 1                      # skip floating_base_joint
            self._joint_limits_low[i]  = self.model.jnt_range[jnt_id, 0]
            self._joint_limits_high[i] = self.model.jnt_range[jnt_id, 1]

        # ── original torso mass  (for domain-randomisation reset) ──
        self._torso_body_id = self.model.body("torso_link").id
        self._original_torso_mass = float(self.model.body_mass[self._torso_body_id])

        # ── mutable env state  (initialised properly in reset_model) ──
        self._step_count       = 0
        self._curriculum_level = 0.0
        self._last_action      = np.zeros(29, dtype=np.float64)
        self._prev_qvel_joints = np.zeros(29, dtype=np.float64)
        self._cmd_vel          = np.zeros(3,  dtype=np.float64)
        self._platform_vel     = np.zeros(2,  dtype=np.float64)
        self._platform_perturbation = np.zeros(2, dtype=np.float64)
        self._obs_history      = deque(maxlen=HISTORY_LENGTH)

        # ── push / command counters ──
        self._push_countdown       = 500
        self._cmd_resample_countdown = 0
        self._total_tracking_reward  = 0.0

        # ── compute resample interval in steps ──
        self._resample_steps = int(round(CMD_RESAMPLE_TIME / self.dt))

    # ───────────────────── body / geom ID cache ─────────────────────
    def _init_body_and_geom_ids(self):
        """Build sets of geom IDs for contact detection."""
        self._platform_geom_id = self.model.geom("platform").id
        self._left_foot_body_id  = self.model.body("left_ankle_roll_link").id
        self._right_foot_body_id = self.model.body("right_ankle_roll_link").id

        pelvis_id = self.model.body("pelvis").id

        # Collect all body IDs in the robot kinematic tree
        robot_body_ids = set()
        for i in range(self.model.nbody):
            bid = i
            while bid != 0:
                if bid == pelvis_id:
                    robot_body_ids.add(i)
                    break
                bid = self.model.body_parentid[bid]

        self._left_foot_geom_ids  = set()
        self._right_foot_geom_ids = set()
        self._robot_collision_geom_ids = set()

        for i in range(self.model.ngeom):
            bid = self.model.geom_bodyid[i]
            # Skip visual-only geoms (contype==0 AND conaffinity==0)
            if self.model.geom_contype[i] == 0 and self.model.geom_conaffinity[i] == 0:
                continue
            if bid not in robot_body_ids:
                continue
            self._robot_collision_geom_ids.add(i)
            if bid == self._left_foot_body_id:
                self._left_foot_geom_ids.add(i)
            elif bid == self._right_foot_body_id:
                self._right_foot_geom_ids.add(i)

        self._foot_geom_ids = self._left_foot_geom_ids | self._right_foot_geom_ids
        self._non_foot_robot_geom_ids = self._robot_collision_geom_ids - self._foot_geom_ids

    # ═══════════════════════════════════════════════════════════════════
    #   OBSERVATIONS
    # ═══════════════════════════════════════════════════════════════════

    def _get_obs_frame(self):
        """Build one 99-dim observation frame  (noise → scale pipeline)."""
        pelvis_xmat = self.data.body("pelvis").xmat.reshape(3, 3)

        # ── raw values ──
        platform_vel_world = np.array([
            self.data.qvel[35], self.data.qvel[36], 0.0,
        ])
        rel_vel_world = self.data.qvel[0:3] - platform_vel_world
        body_lin_vel  = pelvis_xmat.T @ rel_vel_world

        body_ang_vel       = pelvis_xmat.T @ self.data.qvel[3:6]
        projected_gravity  = pelvis_xmat.T @ np.array([0.0, 0.0, -1.0])
        joint_pos_rel      = self.data.qpos[7:36] - STANDING_POSE
        joint_vel_rel      = self.data.qvel[6:35].copy()

        # ── add noise to RAW values (Isaac Lab: noise applied before scale) ──
        body_lin_vel      += self.np_random.uniform(-NOISE_LIN_VEL,   NOISE_LIN_VEL,   size=3)
        body_ang_vel      += self.np_random.uniform(-NOISE_ANG_VEL,   NOISE_ANG_VEL,   size=3)
        projected_gravity += self.np_random.uniform(-NOISE_GRAVITY,   NOISE_GRAVITY,    size=3)
        joint_pos_rel     += self.np_random.uniform(-NOISE_JOINT_POS, NOISE_JOINT_POS,  size=29)
        joint_vel_rel     += self.np_random.uniform(-NOISE_JOINT_VEL, NOISE_JOINT_VEL,  size=29)

        # ── apply scale ──
        body_lin_vel  *= SCALE_LIN_VEL
        body_ang_vel  *= SCALE_ANG_VEL        # ×0.2
        joint_vel_rel *= SCALE_JOINT_VEL       # ×0.05

        return np.concatenate([
            body_lin_vel,           # 3
            body_ang_vel,           # 3
            projected_gravity,      # 3
            self._cmd_vel,          # 3   (vx_cmd, vy_cmd, ωz_cmd)
            joint_pos_rel,          # 29
            joint_vel_rel,          # 29
            self._last_action,      # 29  (action applied on the PREVIOUS step)
        ], dtype=np.float64)        # total = 99

    def _get_stacked_obs(self):
        """Return the full 480-dim stacked observation."""
        return np.concatenate(list(self._obs_history), dtype=np.float64)

    # ═══════════════════════════════════════════════════════════════════
    #   VELOCITY COMMANDS  &  CURRICULUM
    # ═══════════════════════════════════════════════════════════════════

    def _lerp_range(self, init_range, limit_range, t):
        """Linear interpolation between two (lo, hi) tuples."""
        lo = init_range[0] + t * (limit_range[0] - init_range[0])
        hi = init_range[1] + t * (limit_range[1] - init_range[1])
        return (lo, hi)

    def _sample_commands(self):
        """Sample new velocity commands based on current curriculum level."""
        t = self._curriculum_level

        # 2% chance of zero command (standing practice)
        if self.np_random.random() < STANDING_PROBABILITY:
            self._cmd_vel = np.zeros(3, dtype=np.float64)
            return

        vx_range = self._lerp_range(CMD_VX_INIT, CMD_VX_LIMIT, t)
        vy_range = self._lerp_range(CMD_VY_INIT, CMD_VY_LIMIT, t)
        wz_range = self._lerp_range(CMD_WZ_INIT, CMD_WZ_LIMIT, t)

        self._cmd_vel = np.array([
            self.np_random.uniform(*vx_range),
            self.np_random.uniform(*vy_range),
            self.np_random.uniform(*wz_range),
        ], dtype=np.float64)

    def _sample_platform_perturbation(self):
        """Sample a new platform perturbation velocity (curriculum-controlled)."""
        pert_max = self._curriculum_level * PLATFORM_VEL_LIMIT
        self._platform_perturbation = np.array([
            self.np_random.uniform(-pert_max, pert_max),
            self.np_random.uniform(-pert_max, pert_max),
        ], dtype=np.float64)

    def _update_curriculum(self):
        """Raise or lower curriculum level based on the episode that just ended."""
        if self._step_count < 10:
            return                      # too short to evaluate
        survival = self._step_count / MAX_EPISODE_STEPS
        avg_track = self._total_tracking_reward / max(self._step_count, 1)

        if survival >= CURRICULUM_SURVIVAL_THRESHOLD and avg_track >= CURRICULUM_TRACKING_THRESHOLD:
            self._curriculum_level = min(1.0, self._curriculum_level + CURRICULUM_STEP_UP)
        elif survival < 0.3:
            self._curriculum_level = max(0.0, self._curriculum_level - CURRICULUM_STEP_DOWN)

    # ═══════════════════════════════════════════════════════════════════
    #   CONTACT DETECTION
    # ═══════════════════════════════════════════════════════════════════

    def _detect_contacts(self):
        """Returns (left_foot_contact, right_foot_contact, n_undesired)."""
        left  = False
        right = False
        undesired = 0

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist > 0:
                continue                # not penetrating → no real contact
            g1, g2 = c.geom1, c.geom2
            geom_pair = {g1, g2}

            if self._platform_geom_id not in geom_pair:
                continue                # not touching the platform
            other = (geom_pair - {self._platform_geom_id}).pop()

            if other in self._left_foot_geom_ids:
                left = True
            elif other in self._right_foot_geom_ids:
                right = True
            elif other in self._non_foot_robot_geom_ids:
                undesired += 1

        return left, right, undesired

    # ═══════════════════════════════════════════════════════════════════
    #   GAIT  &  FEET  REWARDS
    # ═══════════════════════════════════════════════════════════════════

    def _compute_gait_reward(self, left_contact, right_contact):
        """Phase-based alternating-foot gait reward.

        Only active when the velocity command is non-trivial.
        """
        cmd_speed = np.linalg.norm(self._cmd_vel[:2])
        if cmd_speed < 0.05:
            return 0.0                  # standing → no gait reward

        phase = (self._step_count * self.dt / GAIT_PERIOD) % 1.0

        # Left foot: stance when phase < threshold
        left_desired_stance  = phase < GAIT_THRESHOLD
        # Right foot: offset by 0.5 (half cycle)
        right_phase = (phase - GAIT_OFFSETS[1]) % 1.0
        right_desired_stance = right_phase < GAIT_THRESHOLD

        left_match  = float(left_desired_stance  == left_contact)
        right_match = float(right_desired_stance == right_contact)

        return (left_match + right_match) / 2.0

    def _compute_feet_clearance(self, left_contact, right_contact,
                                left_foot_vel, right_foot_vel):
        """Reward swing foot for reaching target clearance height."""
        reward = 0.0

        if not left_contact:
            h = self.data.body("left_ankle_roll_link").xpos[2]
            err = h - FOOT_CLEARANCE_TARGET
            gauss = np.exp(-err**2 / (2.0 * FOOT_CLEARANCE_STD**2))
            speed = np.linalg.norm(left_foot_vel)
            reward += gauss * np.tanh(FOOT_CLEARANCE_TANH_MULT * speed)

        if not right_contact:
            h = self.data.body("right_ankle_roll_link").xpos[2]
            err = h - FOOT_CLEARANCE_TARGET
            gauss = np.exp(-err**2 / (2.0 * FOOT_CLEARANCE_STD**2))
            speed = np.linalg.norm(right_foot_vel)
            reward += gauss * np.tanh(FOOT_CLEARANCE_TANH_MULT * speed)

        return reward

    # ═══════════════════════════════════════════════════════════════════
    #   DOMAIN  RANDOMISATION
    # ═══════════════════════════════════════════════════════════════════

    def _randomise_domain(self):
        """Called at the start of each episode."""
        # ── friction ──
        friction = self.np_random.uniform(*FRICTION_RANGE)
        self.model.geom_friction[self._platform_geom_id, 0] = friction
        for gid in self._foot_geom_ids:
            self.model.geom_friction[gid, 0] = friction

        # ── torso mass offset  (Isaac Lab: add_base_mass −1 … +3 kg) ──
        mass_offset = self.np_random.uniform(-1.0, 3.0)
        self.model.body_mass[self._torso_body_id] = self._original_torso_mass + mass_offset
        mujoco.mj_setConst(self.model, self.data)

    def _maybe_push(self):
        """Velocity-impulse push  (Isaac Lab: push_by_setting_velocity)."""
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            self.data.qvel[0] += self.np_random.uniform(*PUSH_VEL_RANGE)
            self.data.qvel[1] += self.np_random.uniform(*PUSH_VEL_RANGE)
            self._push_countdown = self.np_random.integers(*PUSH_INTERVAL_RANGE)

    # ═══════════════════════════════════════════════════════════════════
    #   RESET
    # ═══════════════════════════════════════════════════════════════════

    def reset_model(self):
        # ── curriculum update  (evaluates the episode that just finished) ──
        self._update_curriculum()

        # ── domain randomisation ──
        self._randomise_domain()

        # ── initial state ──
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[2] = STANDING_HEIGHT
        qpos[3] = 1.0                          # quaternion w
        qpos[7:36] = STANDING_POSE.copy()
        qpos[7:36] += self.np_random.uniform(-0.02, 0.02, size=29)
        # platform at origin
        qpos[36] = 0.0
        qpos[37] = 0.0

        qvel = np.zeros(self.model.nv, dtype=np.float64)
        qvel[6:35] = self.np_random.uniform(-0.01, 0.01, size=29)

        self.set_state(qpos, qvel)
        self.data.xfrc_applied[:] = 0.0

        # ── counters ──
        self._step_count            = 0
        self._total_tracking_reward = 0.0
        self._push_countdown = self.np_random.integers(*PUSH_INTERVAL_RANGE)
        self._cmd_resample_countdown = 0        # sample immediately

        # ── state buffers ──
        self._last_action      = np.zeros(29, dtype=np.float64)
        self._prev_qvel_joints = np.zeros(29, dtype=np.float64)
        self._platform_vel     = np.zeros(2,  dtype=np.float64)

        # ── sample initial commands + platform perturbation ──
        self._sample_commands()
        self._sample_platform_perturbation()

        # ── fill observation history with first frame ──
        self._obs_history.clear()
        first_frame = self._get_obs_frame()
        for _ in range(HISTORY_LENGTH):
            self._obs_history.append(first_frame.copy())

        return self._get_stacked_obs()

    # ═══════════════════════════════════════════════════════════════════
    #   STEP
    # ═══════════════════════════════════════════════════════════════════

    def step(self, action):
        self._step_count += 1

        # ── save previous state for rate penalties ──
        prev_action     = self._last_action.copy()
        prev_qvel       = self._prev_qvel_joints.copy()
        prev_left_pos   = self.data.body("left_ankle_roll_link").xpos.copy()
        prev_right_pos  = self.data.body("right_ankle_roll_link").xpos.copy()

        # ── maybe push ──
        self._maybe_push()

        # ── maybe resample velocity commands ──
        self._cmd_resample_countdown -= 1
        if self._cmd_resample_countdown <= 0:
            self._sample_commands()
            self._sample_platform_perturbation()
            self._cmd_resample_countdown = self._resample_steps

        # ── PD control → torques ──
        action_f64 = action.astype(np.float64)
        target_q = STANDING_POSE + ACTION_SCALE * action_f64

        q_joints  = self.data.qpos[7:36]
        qd_joints = self.data.qvel[6:35]

        torque = kp * (target_q - q_joints) - kd * qd_joints
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

        # ── platform velocity  (centering + curriculum perturbation) ──
        # The platform acts as a moving "bus floor".
        # Perturbation is the curriculum-controlled disturbance.
        # No treadmill — the robot must stay on the platform via natural walking.
        target_platform = self._platform_perturbation.copy()
        max_delta = PLATFORM_ACCEL_MAX * self.dt
        self._platform_vel += np.clip(
            target_platform - self._platform_vel, -max_delta, max_delta,
        )

        # ── simulate ──
        full_ctrl = np.concatenate([torque, self._platform_vel])
        self.do_simulation(full_ctrl, self.frame_skip)

        # ── derived quantities after simulation ──
        pelvis_xmat   = self.data.body("pelvis").xmat.reshape(3, 3)
        body_ang_vel  = pelvis_xmat.T @ self.data.qvel[3:6]

        # Velocity relative to platform in body frame
        platform_vel_world = np.array([
            self.data.qvel[35], self.data.qvel[36], 0.0,
        ])
        rel_vel_world = self.data.qvel[0:3] - platform_vel_world
        body_vel      = pelvis_xmat.T @ rel_vel_world

        projected_gravity = pelvis_xmat.T @ np.array([0.0, 0.0, -1.0])

        qd_joints_now = self.data.qvel[6:35].copy()
        q_joints_now  = self.data.qpos[7:36].copy()

        # Foot velocities  (finite-difference)
        left_foot_vel  = (self.data.body("left_ankle_roll_link").xpos  - prev_left_pos)  / self.dt
        right_foot_vel = (self.data.body("right_ankle_roll_link").xpos - prev_right_pos) / self.dt

        # Contact detection
        left_contact, right_contact, n_undesired = self._detect_contacts()

        # ════════════════════  REWARD  ════════════════════
        # 1. Velocity tracking  (core walking signal)
        vel_error_xy = body_vel[:2] - self._cmd_vel[:2]
        track_lin_vel = np.exp(-np.sum(vel_error_xy**2) / TRACKING_SIGMA_SQ)

        ang_vel_error = body_ang_vel[2] - self._cmd_vel[2]
        track_ang_vel = np.exp(-ang_vel_error**2 / TRACKING_SIGMA_SQ)

        # 2. Alive
        alive = 1.0

        # 3. Regularisation penalties
        lin_vel_z_l2   = body_vel[2] ** 2
        ang_vel_xy_l2  = np.sum(body_ang_vel[:2] ** 2)
        joint_vel_l2   = np.sum(qd_joints_now ** 2)

        joint_acc      = (qd_joints_now - prev_qvel) / self.dt
        joint_acc_l2   = np.sum(joint_acc ** 2)

        action_rate_l2 = np.sum((action_f64 - prev_action) ** 2)

        # Joint-position-limits penalty (violating limits)
        below = np.clip(self._joint_limits_low  - q_joints_now, 0.0, None)
        above = np.clip(q_joints_now - self._joint_limits_high, 0.0, None)
        dof_pos_limits = np.sum(below + above)

        # Energy
        energy = np.sum(np.abs(torque * qd_joints_now))

        # 4. Joint deviation  (L1, grouped)
        q_dev = q_joints_now - STANDING_POSE
        arm_dev   = np.sum(np.abs(q_dev[ARM_JOINT_INDICES]))
        waist_dev = np.sum(np.abs(q_dev[WAIST_JOINT_INDICES]))
        leg_dev   = np.sum(np.abs(q_dev[LEG_DEV_JOINT_INDICES]))

        # 5. Orientation & height
        flat_orientation_l2 = np.sum(projected_gravity[:2] ** 2)
        height              = self.data.qpos[2]
        base_height_l2      = (height - TARGET_HEIGHT) ** 2

        # 6. Gait rewards
        gait_reward = self._compute_gait_reward(left_contact, right_contact)

        # Feet slide  (penalise foot velocity when in contact, relative to platform)
        platform_vel_xy = np.array([self.data.qvel[35], self.data.qvel[36]])
        feet_slide = 0.0
        if left_contact:
            feet_slide += np.sum((left_foot_vel[:2] - platform_vel_xy) ** 2)
        if right_contact:
            feet_slide += np.sum((right_foot_vel[:2] - platform_vel_xy) ** 2)

        # Feet clearance
        feet_clearance = self._compute_feet_clearance(
            left_contact, right_contact, left_foot_vel, right_foot_vel,
        )

        # 7. Undesired contacts
        undesired_penalty = float(n_undesired)

        # 8. COM drift from platform centre
        pelvis_xy   = self.data.body("pelvis").xpos[:2]
        platform_xy = self.data.body("platform").xpos[:2]
        drift_vec   = pelvis_xy - platform_xy
        com_drift   = np.sum(drift_vec ** 2)

        # ── total reward ──
        reward = (
            WEIGHT_TRACK_LIN_VEL      * track_lin_vel
            + WEIGHT_TRACK_ANG_VEL    * track_ang_vel
            + WEIGHT_ALIVE            * alive
            + WEIGHT_LIN_VEL_Z        * lin_vel_z_l2
            + WEIGHT_ANG_VEL_XY       * ang_vel_xy_l2
            + WEIGHT_JOINT_VEL        * joint_vel_l2
            + WEIGHT_JOINT_ACC        * joint_acc_l2
            + WEIGHT_ACTION_RATE      * action_rate_l2
            + WEIGHT_DOF_POS_LIMITS   * dof_pos_limits
            + WEIGHT_ENERGY           * energy
            + WEIGHT_JOINT_DEV_ARMS   * arm_dev
            + WEIGHT_JOINT_DEV_WAIST  * waist_dev
            + WEIGHT_JOINT_DEV_LEGS   * leg_dev
            + WEIGHT_FLAT_ORIENTATION * flat_orientation_l2
            + WEIGHT_BASE_HEIGHT      * base_height_l2
            + WEIGHT_GAIT             * gait_reward
            + WEIGHT_FEET_SLIDE       * feet_slide
            + WEIGHT_FEET_CLEARANCE   * feet_clearance
            + WEIGHT_UNDESIRED_CONTACTS * undesired_penalty
            + WEIGHT_COM_DRIFT        * com_drift
        )

        # ── curriculum tracking metric ──
        self._total_tracking_reward += track_lin_vel

        # ════════════════════  TERMINATION  ════════════════════
        upright_cos = float(self.data.body("pelvis").xmat[8])   # cos(tilt)
        drift_max   = float(np.max(np.abs(drift_vec)))

        terminated = bool(
            height      < TERM_MIN_HEIGHT
            or upright_cos < np.cos(TERM_MAX_TILT)
            or drift_max   > TERM_MAX_DRIFT
        )
        truncated = bool(self._step_count >= MAX_EPISODE_STEPS)

        if terminated and not truncated:
            reward += WEIGHT_TERMINATE

        # ── update state buffers ──
        self._last_action      = action_f64.copy()
        self._prev_qvel_joints = qd_joints_now.copy()

        # ── observation ──
        frame = self._get_obs_frame()
        self._obs_history.append(frame)
        obs = self._get_stacked_obs()

        info = {
            "curriculum_level": self._curriculum_level,
            "cmd_vel":          self._cmd_vel.copy(),
            "track_lin_vel":    track_lin_vel,
            "track_ang_vel":    track_ang_vel,
            "drift":            drift_max,
        }

        return (obs, float(reward), terminated, truncated, info)


# ═══════════════════════════════════════════════════════════════════════
#   Factory for SubprocVecEnv
# ═══════════════════════════════════════════════════════════════════════

def make_walk_env(rank: int):
    """Return a callable that creates a seeded G1WalkEnv."""
    def _init():
        env = G1WalkEnv(render_mode=None)
        env.reset(seed=2000 + rank)
        return env
    return _init
