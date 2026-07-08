"""
Unitree G1 walking environment with curriculum learning on a moving platform.

Closely mirrors the official unitree_rl_mjlab velocity task:
  - Phase-clock observation (sin/cos) for gait timing        → observations.py
  - Alternating-foot gait reward with stance ratio            → rewards.py  feet_gait
  - Exponential velocity tracking reward                      → rewards.py  track_linear_velocity
  - Variable posture tolerance (tight standing, loose walking)→ rewards.py  variable_posture
  - Foot-slip penalty (foot xy vel while in contact)          → rewards.py  feet_slip
  - Action-rate penalty for smooth control                    → velocity_env_cfg.py
  - Velocity command resampling every 3–8 s                   → velocity_command.py
  - Foot contact detection via geom-pair matching             → env_cfgs.py  ContactSensorCfg
  - Asymmetric actor-critic: actor gets 98D, critic gets 115D → velocity_env_cfg.py
  - Step-based velocity curriculum                            → curriculums.py

Adapted for Gymnasium MujocoEnv + SB3 PPO (single-env observations, NumPy ops).
"""

import numpy as np
import mujoco

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from envs.g1_walk_config import (
    MODEL_PATH, OBS_DIM, ACTOR_OBS_DIM,
    TORQUE_LIMITS, ACTION_SCALE, kp, kd,
    STANDING_HEIGHT, STANDING_POSE,
    GAIT_PERIOD, GAIT_OFFSETS, GAIT_STANCE_RATIO,
    VEL_CMD_RESAMPLE_TIME, STANDING_CMD_PROB,
    VEL_TRACKING_SIGMA, ANG_VEL_TRACKING_SIGMA,
    REWARD_VEL_TRACKING, REWARD_GAIT, REWARD_POSTURE,
    REWARD_ANG_VEL_TRACKING, REWARD_STAND_STILL,
    PENALTY_ACTION_RATE, PENALTY_COM_DRIFT, PENALTY_BASE_ANGVEL,
    PENALTY_FOOT_SLIP, PENALTY_TERMINATION,
    PENALTY_BODY_ORIENTATION, PENALTY_JOINT_ACC, PENALTY_JOINT_POS_LIMITS,
    PENALTY_FOOT_CLEARANCE, PENALTY_SOFT_LANDING,
    FOOT_CLEARANCE_TARGET,
    STD_STANDING, STD_WALKING, STD_RUNNING,
    WALKING_THRESHOLD, RUNNING_THRESHOLD,
    CURRICULUM_STAGES, NUM_CURRICULUM_STAGES,
    CURRICULUM_VEL_EXPAND_STEP,
    MIN_HEIGHT, MIN_UPRIGHT,
)


class G1WalkEnv(MujocoEnv):
    """Unitree G1 walking on a moving platform with curriculum learning.

    Observation (115-D flat array):
        Actor features (98D):
            body_ang_vel (3), projected_gravity (3), vel_command (3),
            phase (2), joint_pos_rel (29), joint_vel (29), last_action (29).

        Critic-only features (17D, zeroed by AsymmetricPolicy for actor):
            body_lin_vel (3), platform_vel_body (2), foot_height (2),
            foot_air_time (2), foot_contact (2), foot_contact_forces (6).

    Action (29-D): normalised joint position offsets in [-1, 1].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64,
        )
        super().__init__(
            model_path=MODEL_PATH,
            frame_skip=10,    # 0.002s × 10 = 0.02s control dt (matches official 0.005×4)
            observation_space=observation_space,
            render_mode=render_mode,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

        # ── Internal state ──────────────────────────────────────────────
        self._step_count = 0
        self._total_steps = 0                # global step counter across episodes (for curriculum)
        self._last_action = np.zeros(29, dtype=np.float64)
        self._prev_action = np.zeros(29, dtype=np.float64)
        self._prev_joint_vel = np.zeros(29, dtype=np.float64)

        # ── Foot tracking state ──────────────────────────────────────────
        self._foot_air_time = np.zeros(2, dtype=np.float64)      # [left, right]
        self._foot_contact_time = np.zeros(2, dtype=np.float64)   # [left, right]
        self._prev_foot_contact = np.array([False, False])

        # ── Velocity command ────────────────────────────────────────────
        self._vel_command = np.zeros(3, dtype=np.float64)   # [vx, vy, yaw_rate]
        self._cmd_counter = 0
        self._cmd_resample_steps = 300   # overwritten in _resample_velocity_command

        # ── Platform disturbance ────────────────────────────────────────
        self._platform_vel = np.zeros(2, dtype=np.float64)
        self._platform_target_vel = np.zeros(2, dtype=np.float64)
        self._platform_change_counter = 0

        # ── Curriculum (step-based) ─────────────────────────────────────
        self._curriculum_level = 0
        self._apply_stage(0)

        # ── Foot contact geom IDs ───────────────────────────────────────
        self._setup_foot_contacts()

        # ── Cache body IDs ──────────────────────────────────────────────
        self._torso_body_id = self.model.body("torso_link").id
        self._left_ankle_body_id = self.model.body("left_ankle_roll_link").id
        self._right_ankle_body_id = self.model.body("right_ankle_roll_link").id

    # ═══════════════════════════════════════════════════════════════════════
    # Foot contact detection
    # Follows the official approach: match ankle_roll_link subtree geoms
    # against the platform geom via data.contact pairs.
    # (Official: ContactSensorCfg with subtree pattern for ankle_roll_link)
    # ═══════════════════════════════════════════════════════════════════════

    def _setup_foot_contacts(self):
        """Cache geom IDs of collision-enabled foot geoms + platform."""
        left_body  = self.model.body("left_ankle_roll_link").id
        right_body = self.model.body("right_ankle_roll_link").id
        self._platform_geom_id = self.model.geom("platform").id

        self._left_foot_geom_ids: set[int] = set()
        self._right_foot_geom_ids: set[int] = set()

        for i in range(self.model.ngeom):
            # Skip visual-only geoms (contype=0 AND conaffinity=0)
            if self.model.geom_contype[i] == 0 and self.model.geom_conaffinity[i] == 0:
                continue
            bid = self.model.geom_bodyid[i]
            if bid == left_body:
                self._left_foot_geom_ids.add(i)
            elif bid == right_body:
                self._right_foot_geom_ids.add(i)

    def _detect_foot_contacts(self) -> tuple[bool, bool]:
        """Return (left_contact, right_contact) with the platform."""
        left = False
        right = False
        pid = self._platform_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == pid or g2 == pid:
                other = g2 if g1 == pid else g1
                if other in self._left_foot_geom_ids:
                    left = True
                elif other in self._right_foot_geom_ids:
                    right = True
            if left and right:
                break
        return left, right

    def _get_foot_contact_forces(self) -> np.ndarray:
        """Accumulate contact forces for each foot from MuJoCo contacts.

        Returns shape (2, 3) — [left_force_xyz, right_force_xyz] in world frame.
        Uses data.contact and data.efc_force to compute actual constraint forces.
        """
        forces = np.zeros((2, 3), dtype=np.float64)
        pid = self._platform_geom_id

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == pid or g2 == pid:
                other = g2 if g1 == pid else g1
                foot_idx = -1
                if other in self._left_foot_geom_ids:
                    foot_idx = 0
                elif other in self._right_foot_geom_ids:
                    foot_idx = 1

                if foot_idx >= 0:
                    # Get the constraint force for this contact
                    efc_addr = c.efc_address
                    if efc_addr >= 0 and efc_addr < self.data.nefc:
                        # Normal force
                        normal_force = self.data.efc_force[efc_addr]
                        # Contact frame: rows of c.frame give normal and tangent dirs
                        contact_normal = c.frame[:3]
                        force_world = normal_force * contact_normal

                        # Sign convention: force on foot
                        if g1 == pid:
                            forces[foot_idx] += force_world
                        else:
                            forces[foot_idx] -= force_world

        return forces

    # ═══════════════════════════════════════════════════════════════════════
    # Observation  (mirrors official observations.py actor + critic terms)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_obs(self):
        R = self.data.body("pelvis").xmat.reshape(3, 3)

        # --- Actor features (98D) ---

        # Body-frame angular velocity (qvel[3:6] is in local frame in MuJoCo)
        body_ang_vel = self.data.qvel[3:6].copy()                            # 3

        # Projected gravity  (official: projected_gravity)
        projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])                # 3

        # Velocity command  (official: command)
        vel_command = self._vel_command.copy()                                # 3

        # Phase clock  (official: observations.py → phase())
        t = self._step_count * self.dt
        global_phase = (t % GAIT_PERIOD) / GAIT_PERIOD
        cmd_norm = np.linalg.norm(self._vel_command)
        if cmd_norm > 0.1:
            phase = np.array([
                np.sin(global_phase * 2.0 * np.pi),
                np.cos(global_phase * 2.0 * np.pi),
            ])
        else:
            phase = np.zeros(2)                                               # 2

        # Joint positions relative to default  (official: joint_pos_rel)
        joint_pos = self.data.qpos[7:36] - STANDING_POSE                     # 29

        # Joint velocities  (official: joint_vel_rel)
        joint_vel = self.data.qvel[6:35]                                     # 29

        # Previous action  (official: last_action)
        last_action = self._last_action.copy()                                # 29

        # --- Critic-only features (17D) ---

        # Body-frame linear velocity  (official critic: base_lin_vel)
        body_lin_vel = R.T @ self.data.qvel[0:3]                             # 3

        # Platform velocity in body frame  (our addition for platform env)
        pvel_world = np.array([self.data.qvel[35], self.data.qvel[36], 0.0])
        pvel_body = R.T @ pvel_world                                          # 2 (xy only)

        # Foot height  (official: observations.py → foot_height)
        platform_surface_z = self.data.body("platform").xpos[2] + 0.05
        left_foot_z  = self.data.body("left_ankle_roll_link").xpos[2] - platform_surface_z
        right_foot_z = self.data.body("right_ankle_roll_link").xpos[2] - platform_surface_z
        foot_height = np.array([left_foot_z, right_foot_z])                   # 2

        # Foot air time  (official: observations.py → foot_air_time)
        foot_air_time = self._foot_air_time.copy()                            # 2

        # Foot contact  (official: observations.py → foot_contact)
        left_contact, right_contact = self._detect_foot_contacts()
        foot_contact = np.array([float(left_contact), float(right_contact)])  # 2

        # Foot contact forces  (official: observations.py → foot_contact_forces)
        # Official uses log1p transform: sign(f) * log1p(|f|)
        raw_forces = self._get_foot_contact_forces()  # (2, 3)
        forces_flat = raw_forces.flatten()             # 6
        foot_contact_forces = np.sign(forces_flat) * np.log1p(np.abs(forces_flat))  # 6

        return np.concatenate([
            # Actor features (98D)
            body_ang_vel,        # 3
            projected_gravity,   # 3
            vel_command,         # 3
            phase,               # 2
            joint_pos,           # 29
            joint_vel,           # 29
            last_action,         # 29
            # Critic-only features (17D)
            body_lin_vel,        # 3
            pvel_body[:2],       # 2
            foot_height,         # 2
            foot_air_time,       # 2
            foot_contact,        # 2
            foot_contact_forces, # 6
        ], dtype=np.float64)    # Total: 115

    # ═══════════════════════════════════════════════════════════════════════
    # Velocity command  (official: velocity_command.py → _resample_command)
    # ═══════════════════════════════════════════════════════════════════════

    def _resample_velocity_command(self):
        """Sample a new velocity command within current curriculum bounds."""
        if self.np_random.random() < STANDING_CMD_PROB:
            self._vel_command[:] = 0.0
        else:
            self._vel_command[0] = self.np_random.uniform(*self._vel_x_range)
            self._vel_command[1] = self.np_random.uniform(*self._vel_y_range)
            self._vel_command[2] = self.np_random.uniform(*self._vel_yaw_range)
            # Zero out tiny commands (official: threshold 0.1)
            if np.linalg.norm(self._vel_command) < 0.1:
                self._vel_command[:] = 0.0

        self._cmd_resample_steps = self.np_random.integers(
            int(VEL_CMD_RESAMPLE_TIME[0] / self.dt),
            int(VEL_CMD_RESAMPLE_TIME[1] / self.dt) + 1,
        )
        self._cmd_counter = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Platform disturbance  (simulates bus/train/ship floor)
    # ═══════════════════════════════════════════════════════════════════════

    def _update_platform(self):
        """Periodically change the platform velocity target, then ramp smoothly."""
        self._platform_change_counter -= 1
        if self._platform_change_counter <= 0:
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            mag = self.np_random.uniform(0.0, self._platform_vel_max)
            self._platform_target_vel = np.array([
                mag * np.cos(angle),
                mag * np.sin(angle),
            ])
            self._platform_change_counter = self.np_random.integers(
                int(2.0 / self.dt), int(5.0 / self.dt) + 1,
            )

        max_delta = self._platform_accel_max * self.dt
        diff = self._platform_target_vel - self._platform_vel
        self._platform_vel += np.clip(diff, -max_delta, max_delta)

    # ═══════════════════════════════════════════════════════════════════════
    # Gait reward  (official: rewards.py → feet_gait)
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_gait_reward(self) -> float:
        """Reward for matching the desired alternating foot contact pattern."""
        t = self._step_count * self.dt
        global_phase = (t / GAIT_PERIOD) % 1.0

        left_phase  = (global_phase + GAIT_OFFSETS[0]) % 1.0
        right_phase = (global_phase + GAIT_OFFSETS[1]) % 1.0

        left_should_stance  = left_phase  < GAIT_STANCE_RATIO
        right_should_stance = right_phase < GAIT_STANCE_RATIO

        left_contact, right_contact = self._detect_foot_contacts()

        left_match  = float(left_should_stance  == left_contact)
        right_match = float(right_should_stance == right_contact)

        return (left_match + right_match) / 2.0

    # ═══════════════════════════════════════════════════════════════════════
    # Foot-slip penalty  (official: rewards.py → feet_slip)
    #
    # Official uses site_lin_vel_w (world-frame foot site velocity).
    # We use mj_objectVelocity for the ankle body in world frame.
    # Official does NOT subtract platform velocity.
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_foot_slip(self, left_contact: bool, right_contact: bool) -> float:
        """Penalise foot xy velocity (world frame) while in contact.

        Matches official: vel_xy_norm_sq * in_contact, summed over feet.
        """
        if not left_contact and not right_contact:
            return 0.0

        slip = 0.0
        res = np.zeros(6)   # [ang_vel(3), lin_vel(3)]

        if left_contact:
            mujoco.mj_objectVelocity(
                self.model, self.data,
                mujoco.mjtObj.mjOBJ_XBODY, self._left_ankle_body_id, res, 0,
            )
            slip += float(np.sum(res[3:5] ** 2))

        if right_contact:
            mujoco.mj_objectVelocity(
                self.model, self.data,
                mujoco.mjtObj.mjOBJ_XBODY, self._right_ankle_body_id, res, 0,
            )
            slip += float(np.sum(res[3:5] ** 2))

        return slip

    # ═══════════════════════════════════════════════════════════════════════
    # Curriculum — step-based velocity expansion
    #
    # Mirrors official commands_vel() in curriculums.py:
    #   stage 0 (< CURRICULUM_VEL_EXPAND_STEP):  moderate vel range
    #   stage 1 (≥ CURRICULUM_VEL_EXPAND_STEP):  full vel range
    #
    # Platform disturbance also ramps with stage.
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_stage(self, stage: int):
        """Apply params for *stage*."""
        stage = max(0, min(stage, NUM_CURRICULUM_STAGES - 1))
        cfg = CURRICULUM_STAGES[stage]
        self._curriculum_level    = stage
        self._vel_x_range        = cfg["vel_x_range"]
        self._vel_y_range        = cfg["vel_y_range"]
        self._vel_yaw_range      = cfg["vel_yaw_range"]
        self._platform_vel_max   = cfg["platform_vel_max"]
        self._platform_accel_max = cfg["platform_accel_max"]
        self._max_episode_steps  = cfg["max_episode_steps"]

    def _update_curriculum(self):
        """Check if we should expand velocity ranges based on total step count."""
        if self._total_steps >= CURRICULUM_VEL_EXPAND_STEP and self._curriculum_level < 1:
            self._apply_stage(1)

    def set_curriculum_stage(self, stage: int):
        """Force a specific stage (for evaluation / run_walk.py)."""
        self._apply_stage(stage)
        self._resample_velocity_command()

    def get_curriculum_level(self) -> int:
        """Return the current curriculum level (for logging)."""
        return self._curriculum_level

    # ═══════════════════════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════════════════════

    def reset_model(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = STANDING_HEIGHT
        qpos[3] = 1.0                          # quaternion w
        qpos[7:36] = STANDING_POSE.copy()
        qpos[7:36] += self.np_random.uniform(-0.02, 0.02, size=29)

        qvel = np.zeros(self.model.nv)
        qvel[6:35] = self.np_random.uniform(-0.01, 0.01, size=29)

        self.set_state(qpos, qvel)

        self._step_count = 0
        self._last_action[:] = 0.0
        self._prev_action[:] = 0.0
        self._prev_joint_vel[:] = 0.0
        self._prev_foot_contact[:] = False
        self._foot_air_time[:] = 0.0
        self._foot_contact_time[:] = 0.0
        self._platform_vel[:] = 0.0
        self._platform_target_vel[:] = 0.0
        self._platform_change_counter = self.np_random.integers(
            int(2.0 / self.dt), int(5.0 / self.dt) + 1,
        )
        self._resample_velocity_command()

        # Update curriculum based on total steps
        self._update_curriculum()

        return self._get_obs()

    # ═══════════════════════════════════════════════════════════════════════
    # Step
    # ═══════════════════════════════════════════════════════════════════════

    def step(self, action):
        self._step_count += 1
        self._total_steps += 1
        self._prev_action = self._last_action.copy()
        self._last_action = action.astype(np.float64).copy()

        # ── PD position controller ──────────────────────────────────────
        target_q = STANDING_POSE + ACTION_SCALE * action
        q  = self.data.qpos[7:36]      # joint positions  (before sim)
        qd = self.data.qvel[6:35]      # joint velocities (before sim)
        torque = kp * (target_q - q) - kd * qd
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

        # ── Platform disturbance ────────────────────────────────────────
        self._update_platform()

        # ── Simulate ────────────────────────────────────────────────────
        full_ctrl = np.concatenate([torque, self._platform_vel])
        self.do_simulation(full_ctrl, self.frame_skip)

        # ── Velocity command update ─────────────────────────────────────
        self._cmd_counter += 1
        if self._cmd_counter >= self._cmd_resample_steps:
            self._resample_velocity_command()

        # ── Foot contact detection ──────────────────────────────────────
        left_contact, right_contact = self._detect_foot_contacts()
        contacts = np.array([left_contact, right_contact])

        # Update air time / contact time tracking
        # (official: ContactSensor tracks current_air_time / current_contact_time)
        for i in range(2):
            if contacts[i]:
                self._foot_contact_time[i] += self.dt
                self._foot_air_time[i] = 0.0
            else:
                self._foot_air_time[i] += self.dt
                self._foot_contact_time[i] = 0.0

        # ── Observation  (post-simulation state) ────────────────────────
        obs = self._get_obs()

        # ═════════════════════════════════════════════════════════════════
        # Rewards  (matching official velocity_env_cfg.py exactly)
        # ═════════════════════════════════════════════════════════════════
        R_pelvis = self.data.body("pelvis").xmat.reshape(3, 3)
        R_torso  = self.data.body("torso_link").xmat.reshape(3, 3)
        height   = self.data.qpos[2]
        upright  = float(self.data.body("pelvis").xmat[8])      # R[2,2]
        q_now    = self.data.qpos[7:36]
        qd_now   = self.data.qvel[6:35]

        # Speed decomposition matching official (linear + angular, not norm)
        linear_speed = float(np.linalg.norm(self._vel_command[:2]))
        angular_speed = float(abs(self._vel_command[2]))
        total_speed = linear_speed + angular_speed

        # ── positive rewards ────────────────────────────────────────────

        # 1  Velocity tracking  (official: track_linear_velocity)
        #    root_link_lin_vel_b = body-frame linear velocity
        body_lin_vel = R_pelvis.T @ self.data.qvel[0:3]
        xy_err = float(np.sum((self._vel_command[:2] - body_lin_vel[:2]) ** 2))
        z_err  = float(body_lin_vel[2] ** 2)
        r_vel_track = REWARD_VEL_TRACKING * np.exp(
            -(xy_err + 2.0 * z_err) / VEL_TRACKING_SIGMA
        )

        # 2  Angular velocity tracking (official: track_angular_velocity)
        #    root_link_ang_vel_b = body-frame angular velocity
        base_ang_vel = self.data.qvel[3:6].copy()
        z_err_ang = float((self._vel_command[2] - base_ang_vel[2]) ** 2)
        xy_err_ang = float(np.sum(base_ang_vel[:2] ** 2))
        r_ang_vel_track = REWARD_ANG_VEL_TRACKING * np.exp(
            -(z_err_ang + 0.05 * xy_err_ang) / ANG_VEL_TRACKING_SIGMA
        )

        # 3  Gait  (official: feet_gait — only when moving)
        r_gait = REWARD_GAIT * self._compute_gait_reward() if total_speed > 0.1 else 0.0

        # 4  Variable posture  (official: variable_posture — 3 speed regimes)
        if total_speed < WALKING_THRESHOLD:
            std = STD_STANDING
        elif total_speed < RUNNING_THRESHOLD:
            std = STD_WALKING
        else:
            std = STD_RUNNING
        posture_err = (q_now - STANDING_POSE) ** 2
        r_posture = REWARD_POSTURE * np.exp(
            -float(np.mean(posture_err / (std ** 2)))
        )

        # ── negative penalties ──────────────────────────────────────────

        # 5  Body orientation L2 (official: body_orientation_l2 on torso_link)
        projected_gravity_torso = R_torso.T @ np.array([0.0, 0.0, -1.0])
        p_body_orientation = PENALTY_BODY_ORIENTATION * float(np.sum(projected_gravity_torso[:2] ** 2))

        # 6  Body angular velocity (official: body_angular_velocity_penalty on torso_link)
        #    Uses world-frame angular velocity of torso, xy only
        torso_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_XBODY, self._torso_body_id, torso_vel, 0,
        )
        torso_ang_vel_w = torso_vel[:3]  # world-frame angular velocity
        p_base_angvel = PENALTY_BASE_ANGVEL * float(np.sum(torso_ang_vel_w[:2] ** 2))

        # 7  Action rate  (official: action_rate_l2)
        p_action_rate = PENALTY_ACTION_RATE * float(
            np.sum((self._last_action - self._prev_action) ** 2)
        )

        # 8  Joint acceleration
        p_joint_acc = PENALTY_JOINT_ACC * float(
            np.sum(((qd_now - self._prev_joint_vel) / self.dt) ** 2)
        )

        # 9  Joint position limits
        q_lower = self.model.jnt_range[1:30, 0]
        q_upper = self.model.jnt_range[1:30, 1]
        out_of_limits = np.maximum(0.0, q_now - q_upper) + np.maximum(0.0, q_lower - q_now)
        p_joint_limits = PENALTY_JOINT_POS_LIMITS * float(np.sum(out_of_limits))

        # 10 COM drift relative to platform (platform-specific, not in official)
        pelvis_xy   = self.data.body("pelvis").xpos[:2]
        platform_xy = self.data.body("platform").xpos[:2]
        drift       = pelvis_xy - platform_xy
        p_com_drift = PENALTY_COM_DRIFT * float(np.sum(drift ** 2))

        # 11 Foot slip (official: feet_slip — world frame, no platform subtraction)
        p_foot_slip = (
            PENALTY_FOOT_SLIP * self._compute_foot_slip(left_contact, right_contact)
            if total_speed > 0.1 else 0.0
        )

        # 12 Foot clearance (official: feet_clearance)
        p_foot_clearance = 0.0
        if total_speed > 0.1:
            platform_surface_z = self.data.body("platform").xpos[2] + 0.05
            left_z  = self.data.body("left_ankle_roll_link").xpos[2] - platform_surface_z
            right_z = self.data.body("right_ankle_roll_link").xpos[2] - platform_surface_z

            left_vel = np.zeros(6)
            right_vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self._left_ankle_body_id, left_vel, 0)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self._right_ankle_body_id, right_vel, 0)

            # Official: delta = |foot_z - target|, cost = sum(delta * vel_norm)
            if not left_contact:
                left_speed = float(np.linalg.norm(left_vel[3:5]))
                p_foot_clearance += float(abs(left_z - FOOT_CLEARANCE_TARGET) * left_speed)
            if not right_contact:
                right_speed = float(np.linalg.norm(right_vel[3:5]))
                p_foot_clearance += float(abs(right_z - FOOT_CLEARANCE_TARGET) * right_speed)

        p_foot_clearance *= PENALTY_FOOT_CLEARANCE

        # 13 Soft landing (official: force magnitude at first contact)
        p_soft_landing = 0.0
        if total_speed > 0.1:
            # first_contact = currently in contact AND was NOT in contact last step
            first_left  = left_contact  and not self._prev_foot_contact[0]
            first_right = right_contact and not self._prev_foot_contact[1]

            if first_left or first_right:
                contact_forces = self._get_foot_contact_forces()  # (2, 3)
                if first_left:
                    p_soft_landing += float(np.linalg.norm(contact_forces[0]))
                if first_right:
                    p_soft_landing += float(np.linalg.norm(contact_forces[1]))

        p_soft_landing *= PENALTY_SOFT_LANDING

        # 14 Stand still (official: joint deviation² when standing)
        p_stand_still = (
            REWARD_STAND_STILL * float(np.sum((q_now - STANDING_POSE) ** 2))
            if total_speed <= 0.1 else 0.0
        )

        # ── total ───────────────────────────────────────────────────────
        reward = (
            r_vel_track + r_ang_vel_track + r_gait + r_posture
            + p_body_orientation + p_base_angvel + p_action_rate
            + p_joint_acc + p_joint_limits + p_com_drift
            + p_foot_slip + p_foot_clearance + p_soft_landing + p_stand_still
        )

        # Update previous states
        self._prev_joint_vel = qd_now.copy()
        self._prev_foot_contact = contacts.copy()

        # ── termination  (official: bad_orientation at 70°) ─────────────
        terminated = bool(height < MIN_HEIGHT or upright < MIN_UPRIGHT)
        if terminated:
            reward += PENALTY_TERMINATION

        truncated = bool(self._step_count >= self._max_episode_steps)
        info = {"curriculum_level": self._curriculum_level}

        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════════
# Factory for SubprocVecEnv
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(rank: int):
    def _init():
        env = G1WalkEnv(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init
