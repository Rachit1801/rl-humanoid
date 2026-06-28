"""
Unitree G1 walking environment with curriculum learning on a moving platform.

Key features adopted from the official unitree_rl_mjlab code:
  - Phase-clock observation (sin/cos) for gait timing        → observations.py
  - Alternating-foot gait reward with stance ratio            → rewards.py  feet_gait
  - Exponential velocity tracking reward                      → rewards.py  track_linear_velocity
  - Variable posture tolerance (tight standing, loose walking)→ rewards.py  variable_posture
  - Foot-slip penalty (foot xy vel while in contact)          → rewards.py  feet_slip
  - Action-rate penalty for smooth control                    → velocity_env_cfg.py
  - Velocity command resampling every 3–8 s                   → velocity_command.py
  - Foot contact detection via geom-pair matching             → env_cfgs.py  ContactSensorCfg

Adapted for Gymnasium MujocoEnv + SB3 PPO (single-env observations, NumPy ops).
"""

import numpy as np
import mujoco

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from envs.g1_walk_config import (
    MODEL_PATH, OBS_DIM,
    TORQUE_LIMITS, ACTION_SCALE, kp, kd,
    STANDING_HEIGHT, STANDING_POSE,
    GAIT_PERIOD, GAIT_OFFSETS, GAIT_STANCE_RATIO,
    VEL_CMD_RESAMPLE_TIME, STANDING_CMD_PROB,
    VEL_TRACKING_SIGMA,
    REWARD_ALIVE, REWARD_HEIGHT, REWARD_UPRIGHT,
    REWARD_VEL_TRACKING, REWARD_GAIT, REWARD_POSTURE,
    REWARD_STAND_STILL,
    PENALTY_ENERGY, PENALTY_JOINT_VEL, PENALTY_ACTION,
    PENALTY_ACTION_RATE, PENALTY_COM_DRIFT, PENALTY_BASE_ANGVEL,
    PENALTY_FOOT_SLIP, PENALTY_TERMINATION,
    HEIGHT_GAUSSIAN_K, MIN_HEIGHT, MIN_UPRIGHT,
    STD_STANDING, STD_WALKING, STD_RUNNING,
    WALKING_THRESHOLD, RUNNING_THRESHOLD,
    CURRICULUM_STAGES, NUM_CURRICULUM_STAGES,
    PROMOTE_SURVIVAL_FRAC, DEMOTE_SURVIVAL_FRAC,
    PROMOTE_STREAK, DEMOTE_STREAK,
)


class G1WalkEnv(MujocoEnv):
    """Unitree G1 walking on a moving platform with curriculum learning.

    Observation (103-D):
        body_ang_vel (3), body_lin_vel (3), projected_gravity (3),
        vel_command (3), phase (2), joint_pos_rel (29), joint_vel (29),
        last_action (29), platform_vel_body (2).

    Action (29-D): normalised joint position offsets in [-1, 1].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64,
        )
        super().__init__(
            model_path=MODEL_PATH,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

        # ── Internal state ──────────────────────────────────────────────
        self._step_count = 0
        self._last_action = np.zeros(29, dtype=np.float64)

        # ── Velocity command ────────────────────────────────────────────
        self._vel_command = np.zeros(3, dtype=np.float64)   # [vx, vy, yaw_rate]
        self._cmd_counter = 0
        self._cmd_resample_steps = 300   # overwritten in _resample_velocity_command

        # ── Platform disturbance ────────────────────────────────────────
        self._platform_vel = np.zeros(2, dtype=np.float64)
        self._platform_target_vel = np.zeros(2, dtype=np.float64)
        self._platform_change_counter = 0

        # ── Adaptive curriculum  (per-env promote / demote) ──────────────
        self._curriculum_level = 0
        self._consecutive_successes = 0
        self._consecutive_failures = 0
        self._episode_reward_accum = 0.0
        self._apply_stage(0)

        # ── Foot contact geom IDs ───────────────────────────────────────
        self._setup_foot_contacts()

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

    # ═══════════════════════════════════════════════════════════════════════
    # Observation  (mirrors official observations.py actor terms)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_obs(self):
        R = self.data.body("pelvis").xmat.reshape(3, 3)

        # Body-frame angular velocity  (official: base_ang_vel)
        body_ang_vel = R.T @ self.data.qvel[3:6]

        # Body-frame linear velocity  (official critic: base_lin_vel)
        body_lin_vel = R.T @ self.data.qvel[0:3]

        # Projected gravity  (official: projected_gravity)
        projected_gravity = R.T @ np.array([0.0, 0.0, -1.0])

        # Velocity command  (official: command)
        vel_command = self._vel_command.copy()

        # Phase clock  (official: observations.py → phase())
        t = self._step_count * self.dt
        global_phase = (t % GAIT_PERIOD) / GAIT_PERIOD
        if np.linalg.norm(self._vel_command) > 0.1:
            phase = np.array([
                np.sin(global_phase * 2.0 * np.pi),
                np.cos(global_phase * 2.0 * np.pi),
            ])
        else:
            phase = np.zeros(2)

        # Joint positions relative to default  (official: joint_pos_rel)
        joint_pos = self.data.qpos[7:36] - STANDING_POSE

        # Joint velocities  (official: joint_vel_rel)
        joint_vel = self.data.qvel[6:35]

        # Previous action  (official: last_action)
        last_action = self._last_action.copy()

        # Platform velocity in body frame  (new: gives the policy floor-motion info)
        pvel_world = np.array([self.data.qvel[35], self.data.qvel[36], 0.0])
        pvel_body = R.T @ pvel_world

        return np.concatenate([
            body_ang_vel,        # 3
            body_lin_vel,        # 3
            projected_gravity,   # 3
            vel_command,         # 3
            phase,               # 2
            joint_pos,           # 29
            joint_vel,           # 29
            last_action,         # 29
            pvel_body[:2],       # 2
        ], dtype=np.float64)    # Total: 103

    # ═══════════════════════════════════════════════════════════════════════
    # Velocity command  (official: velocity_command.py → _resample_command)
    # ═══════════════════════════════════════════════════════════════════════

    def _resample_velocity_command(self):
        """Sample a new velocity command within current curriculum bounds."""
        if self.np_random.random() < STANDING_CMD_PROB:
            self._vel_command[:] = 0.0
        else:
            self._vel_command[0] = self.np_random.uniform(*self._vel_x_range)
            self._vel_command[1] = 0.0    # forward only for now
            self._vel_command[2] = 0.0    # no yaw for now
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
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_foot_slip(self) -> float:
        """Penalise foot xy velocity (relative to platform) while in contact.

        Uses mj_objectVelocity with mjOBJ_XBODY for velocity at the body
        frame origin, which is closer to the foot contact point than
        data.cvel (body center of mass).
        """
        left_contact, right_contact = self._detect_foot_contacts()
        if not left_contact and not right_contact:
            return 0.0

        left_id  = self.model.body("left_ankle_roll_link").id
        right_id = self.model.body("right_ankle_roll_link").id
        pvel_xy  = np.array([self.data.qvel[35], self.data.qvel[36]])
        slip = 0.0
        res = np.zeros(6)   # [ang_vel(3), lin_vel(3)]

        if left_contact:
            mujoco.mj_objectVelocity(
                self.model, self.data,
                mujoco.mjtObj.mjOBJ_XBODY, left_id, res, 0,
            )
            rel_xy = res[3:5] - pvel_xy
            slip += float(np.sum(rel_xy ** 2))

        if right_contact:
            mujoco.mj_objectVelocity(
                self.model, self.data,
                mujoco.mjtObj.mjOBJ_XBODY, right_id, res, 0,
            )
            rel_xy = res[3:5] - pvel_xy
            slip += float(np.sum(rel_xy ** 2))

        return slip

    # ═══════════════════════════════════════════════════════════════════════
    # Adaptive curriculum  — per-env promote / demote
    #
    # Inspired by official terrain_levels_vel() in curriculums.py:
    #   • Robots that walked far enough → move_up to harder terrain
    #   • Robots that fell early        → move_down to easier terrain
    #
    # Here we track survival fraction and consecutive streaks.
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_stage(self, stage: int):
        """Apply params for *stage* without resetting streak counters."""
        stage = max(0, min(stage, NUM_CURRICULUM_STAGES - 1))
        cfg = CURRICULUM_STAGES[stage]
        self._curriculum_level    = stage
        self._vel_x_range        = cfg["vel_x_range"]
        self._platform_vel_max   = cfg["platform_vel_max"]
        self._platform_accel_max = cfg["platform_accel_max"]
        self._max_episode_steps  = cfg["max_episode_steps"]

    def _evaluate_episode(self):
        """Called at episode end: promote/demote based on survival fraction.

        Logic mirrors the official terrain curriculum:
          - move_up  : robot walked far enough  (here: survived long enough)
          - move_down: robot fell early          (here: survived too briefly)
        """
        if self._max_episode_steps == 0:
            return

        survival_frac = self._step_count / self._max_episode_steps

        if survival_frac >= PROMOTE_SURVIVAL_FRAC:
            # Good episode
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            if (self._consecutive_successes >= PROMOTE_STREAK
                    and self._curriculum_level < NUM_CURRICULUM_STAGES - 1):
                self._apply_stage(self._curriculum_level + 1)
                self._consecutive_successes = 0
        elif survival_frac < DEMOTE_SURVIVAL_FRAC:
            # Bad episode — fell very early
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            if (self._consecutive_failures >= DEMOTE_STREAK
                    and self._curriculum_level > 0):
                self._apply_stage(self._curriculum_level - 1)
                self._consecutive_failures = 0
        else:
            # Mediocre — reset both streaks, stay at current level
            self._consecutive_successes = 0
            self._consecutive_failures = 0

    def set_curriculum_stage(self, stage: int):
        """Force a specific stage (for evaluation / run_walk.py)."""
        self._apply_stage(stage)
        self._consecutive_successes = 0
        self._consecutive_failures = 0
        self._resample_velocity_command()

    def get_curriculum_level(self) -> int:
        """Return the current curriculum level (for logging)."""
        return self._curriculum_level

    # ═══════════════════════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════════════════════

    def reset_model(self):
        # ── Evaluate the episode that just ended (promote / demote) ─────
        if self._step_count > 0:
            self._evaluate_episode()

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
        self._episode_reward_accum = 0.0
        self._platform_vel[:] = 0.0
        self._platform_target_vel[:] = 0.0
        self._platform_change_counter = self.np_random.integers(
            int(2.0 / self.dt), int(5.0 / self.dt) + 1,
        )
        self._resample_velocity_command()

        return self._get_obs()

    # ═══════════════════════════════════════════════════════════════════════
    # Step
    # ═══════════════════════════════════════════════════════════════════════

    def step(self, action):
        self._step_count += 1
        prev_action = self._last_action.copy()
        self._last_action = action.astype(np.float64).copy()

        # ── PD position controller  (same as your g1_env.py) ────────────
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

        # ── Observation  (post-simulation state) ────────────────────────
        obs = self._get_obs()

        # ═════════════════════════════════════════════════════════════════
        # Rewards  (post-simulation state for all except energy)
        # ═════════════════════════════════════════════════════════════════
        R = self.data.body("pelvis").xmat.reshape(3, 3)
        height  = self.data.qpos[2]
        upright = float(self.data.body("pelvis").xmat[8])      # R[2,2]
        q_now   = self.data.qpos[7:36]
        # Speed decomposition matching official (linear + angular, not norm)
        linear_speed = float(np.linalg.norm(self._vel_command[:2]))
        angular_speed = float(abs(self._vel_command[2]))
        total_speed = linear_speed + angular_speed

        # ── positive rewards ────────────────────────────────────────────

        # 1  Alive
        r_alive = REWARD_ALIVE

        # 2  Height  (Gaussian)
        r_height = REWARD_HEIGHT * np.exp(
            -HEIGHT_GAUSSIAN_K * (height - STANDING_HEIGHT) ** 2
        )

        # 3  Upright
        r_upright = REWARD_UPRIGHT * max(0.0, upright)

        # 4  Velocity tracking  (official: track_linear_velocity)
        #    Track velocity relative to platform, in body frame.
        body_lin_vel    = R.T @ self.data.qvel[0:3]
        pvel_world      = np.array([self.data.qvel[35], self.data.qvel[36], 0.0])
        pvel_body       = R.T @ pvel_world
        relative_vel    = body_lin_vel - pvel_body
        xy_err = float(np.sum((self._vel_command[:2] - relative_vel[:2]) ** 2))
        z_err  = float(relative_vel[2] ** 2)
        r_vel_track = REWARD_VEL_TRACKING * np.exp(
            -(xy_err + 2.0 * z_err) / VEL_TRACKING_SIGMA
        )

        # 5  Gait  (official: feet_gait — only when moving)
        r_gait = REWARD_GAIT * self._compute_gait_reward() if total_speed > 0.1 else 0.0

        # 6  Variable posture  (official: variable_posture — 3 speed regimes)
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

        # 7  Energy  (pre-sim torque × pre-sim velocity)
        p_energy = PENALTY_ENERGY * float(np.sum(np.abs(torque * qd)))

        # 8  Joint velocity
        p_jvel = PENALTY_JOINT_VEL * float(np.sum(qd ** 2))

        # 9  Action magnitude
        p_action = PENALTY_ACTION * float(np.sum(action ** 2))

        # 10 Action rate  (official: action_rate_l2)
        p_action_rate = PENALTY_ACTION_RATE * float(
            np.sum((self._last_action - prev_action) ** 2)
        )

        # 11 COM drift relative to platform  (your existing design)
        pelvis_xy   = self.data.body("pelvis").xpos[:2]
        platform_xy = self.data.body("platform").xpos[:2]
        drift       = pelvis_xy - platform_xy
        p_com_drift = PENALTY_COM_DRIFT * float(np.sum(drift ** 2))

        # 12 Base angular velocity  (official: body_ang_vel — XY only, exclude Z/yaw)
        p_base_angvel = PENALTY_BASE_ANGVEL * float(
            np.sum(self.data.qvel[3:5] ** 2)
        )

        # 13 Foot slip  (official: feet_slip — only when moving)
        p_foot_slip = (
            PENALTY_FOOT_SLIP * self._compute_foot_slip()
            if total_speed > 0.1 else 0.0
        )

        # 14 Stand still  (official: stand_still — only when standing)
        p_stand_still = (
            REWARD_STAND_STILL * float(np.sum((q_now - STANDING_POSE) ** 2))
            if total_speed <= 0.1 else 0.0
        )

        # ── total ───────────────────────────────────────────────────────
        reward = (
            r_alive + r_height + r_upright + r_vel_track + r_gait + r_posture
            + p_energy + p_jvel + p_action + p_action_rate
            + p_com_drift + p_base_angvel + p_foot_slip + p_stand_still
        )

        # ── termination  (official: bad_orientation at 70°) ─────────────
        terminated = bool(height < MIN_HEIGHT or upright < MIN_UPRIGHT)
        if terminated:
            reward += PENALTY_TERMINATION

        # Accumulate reward for curriculum evaluation
        self._episode_reward_accum += reward

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
