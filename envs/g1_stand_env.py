"""
Unitree G1 29-DOF Humanoid Standing Environment
================================================

A Gymnasium environment that trains the G1 humanoid robot to stand stably
using reinforcement learning with PPO.

Observation Space (67-dim):
    - Base angular velocity in body frame (3)
    - Base linear velocity in body frame (3)
    - Projected gravity vector in body frame (3)
    - Joint positions relative to standing pose (29)
    - Joint velocities (29)

Action Space (29-dim, continuous [-1, 1]):
    - Normalized joint position offsets from standing pose
    - Internally mapped via PD controller to joint torques

Reward:
    - Alive bonus: +2.0 per step
    - Height reward: Gaussian peak at standing height (max +3.0)
    - Upright reward: pelvis z-axis alignment (max +2.0)
    - Energy penalty: penalizes excessive torque x velocity
    - Joint velocity penalty: encourages stillness
    - Action penalty: penalizes large actions
    - Posture penalty: penalizes deviation from standing pose
    - COM drift penalty: penalizes horizontal drift from origin
    - Base angular velocity penalty: prevents rotational drift

Termination:
    - Pelvis height < 0.4m (fallen)
    - Pelvis tilt > ~40deg from upright
    - Episode truncated at 2000 steps
"""

import os
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "scene_29dof.xml"
)

# ─── Robot Constants ──────────────────────────────────────────────────────────

# Default standing height from XML: pelvis pos="0 0 0.793"
STANDING_HEIGHT = 0.793

# Standing pose: all joints at zero (natural standing configuration for G1)
STANDING_POSE = np.zeros(29, dtype=np.float64)

# Per-joint action scale — how far (radians) the policy can deviate each joint
# from the standing pose. Larger = more range of motion for that joint.
# Order: 6 left leg, 6 right leg, 3 waist, 7 left arm, 7 right arm
ACTION_SCALE = np.array([
    # Left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,
    # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,
    # Waist:     yaw, roll, pitch
    0.15, 0.15, 0.15,
    # Left arm:  shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,
    # Right arm: shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,
], dtype=np.float64)

# PD gains from Unitree official config (unitree_rl_lab)
KP = np.array([
    100, 100, 100, 150,  40,  40,   # Left leg
    100, 100, 100, 150,  40,  40,   # Right leg
    200, 200, 200,                   # Waist
     40,  40,  40,  40,  40,  40, 40,  # Left arm
     40,  40,  40,  40,  40,  40, 40,  # Right arm
], dtype=np.float64)

KD = np.array([
     2,  2,  2,  4,  2,  2,   # Left leg
     2,  2,  2,  4,  2,  2,   # Right leg
     5,  5,  5,                # Waist
    10, 10, 10, 10, 10, 10, 10,  # Left arm
    10, 10, 10, 10, 10, 10, 10,  # Right arm
], dtype=np.float64)

# Torque limits from MJCF actuator definitions
TORQUE_LIMITS = np.array([
     88,  88,  88, 139,  50,  50,   # Left leg
     88,  88,  88, 139,  50,  50,   # Right leg
     88,  50,  50,                   # Waist
     25,  25,  25,  25,  25,   5, 5,  # Left arm
     25,  25,  25,  25,  25,   5, 5,  # Right arm
], dtype=np.float64)

# ─── Episode Settings ─────────────────────────────────────────────────────────
MAX_EPISODE_STEPS = 2000      # Doubled: policy must learn long-term stability

# ─── Reward Weights ───────────────────────────────────────────────────────────
# Tuned for stable long-duration standing.
# Max per-step reward ~ 7.0 (standing perfectly still at origin)
REWARD_ALIVE         =  2.0     # Constant survival bonus (dominant signal)
REWARD_HEIGHT        =  3.0     # Gaussian peak at standing height
REWARD_UPRIGHT       =  2.0     # Alignment of pelvis z-axis with world z
PENALTY_ENERGY       = -0.001   # |torque x joint_vel|
PENALTY_JOINT_VEL    = -0.001   # joint_vel^2  (doubled: punish drift harder)
PENALTY_ACTION       = -0.01    # action^2
PENALTY_POSTURE      = -0.15    # (joint_pos - standing_pose)^2  (increased)
PENALTY_COM_DRIFT    = -3.0     # xy position drift from origin (NEW)
PENALTY_BASE_ANGVEL  = -0.1     # base angular velocity^2 (NEW: prevents rotational drift)
HEIGHT_GAUSSIAN_K   = 40.0     # Sharpness of height Gaussian


class G1StandEnv(MujocoEnv):
    """
    Gymnasium environment for the Unitree G1 29-DOF humanoid standing task.

    The robot starts in a standing pose and must learn to maintain balance.
    Uses a PD controller to convert normalized actions into joint torques.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        # Observation: 67-dim vector
        # ang_vel(3) + lin_vel(3) + proj_gravity(3) + joint_pos(29) + joint_vel(29)
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(67,), dtype=np.float64
        )

        super().__init__(
            model_path=MODEL_PATH,
            frame_skip=5,           # Control at 100 Hz (physics at 500 Hz)
            observation_space=observation_space,
            render_mode=render_mode,
        )

        # Override action space to normalized [-1, 1] for all 29 actuators
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(29,), dtype=np.float32
        )

        self._step_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        Build 67-dimensional observation vector.

        All base velocities and gravity are expressed in the pelvis (body)
        frame so the policy is invariant to global yaw orientation.
        """
        # Pelvis rotation matrix (world → body)
        pelvis_xmat = self.data.body("pelvis").xmat.reshape(3, 3)

        # Base angular velocity in body frame
        ang_vel_world = self.data.qvel[3:6]
        ang_vel_body = pelvis_xmat.T @ ang_vel_world          # (3,)

        # Base linear velocity in body frame
        lin_vel_world = self.data.qvel[0:3]
        lin_vel_body = pelvis_xmat.T @ lin_vel_world           # (3,)

        # Projected gravity in body frame
        # When upright: [0, 0, -1]; when tilted: rotates accordingly
        projected_gravity = pelvis_xmat.T @ np.array([0.0, 0.0, -1.0])  # (3,)

        # Joint positions relative to standing pose
        joint_pos = self.data.qpos[7:] - STANDING_POSE         # (29,)

        # Joint velocities
        joint_vel = self.data.qvel[6:]                          # (29,)

        obs = np.concatenate([
            ang_vel_body,        # 3
            lin_vel_body,        # 3
            projected_gravity,   # 3
            joint_pos,           # 29
            joint_vel,           # 29
        ]).astype(np.float64)    # Total: 67

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────────
    def reset_model(self) -> np.ndarray:
        """
        Reset to standing pose with small random perturbations.

        The perturbations break symmetry and improve generalization
        without being large enough to cause immediate falls.
        """
        qpos = np.zeros(self.model.nq)
        qpos[2] = STANDING_HEIGHT       # Pelvis z-height
        qpos[3] = 1.0                   # Quaternion w (identity rotation)
        qpos[7:] = STANDING_POSE.copy()

        # Small random perturbation on joint positions
        qpos[7:] += self.np_random.uniform(-0.02, 0.02, size=29)

        qvel = np.zeros(self.model.nv)
        qvel[6:] = self.np_random.uniform(-0.01, 0.01, size=29)

        self.set_state(qpos, qvel)
        self._step_count = 0

        return self._get_obs()

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Execute one environment step.

        1. Map normalized action → target joint positions
        2. Compute PD torques
        3. Step physics simulation
        4. Compute reward and check termination

        Parameters
        ----------
        action : np.ndarray, shape (29,)
            Normalized joint offsets in [-1, 1].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        self._step_count += 1

        # ── PD Control ────────────────────────────────────────────────────────
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Target = standing pose + scaled action
        target_q = STANDING_POSE + ACTION_SCALE * action

        # Current joint state
        q  = self.data.qpos[7:]     # Joint positions  (29,)
        qd = self.data.qvel[6:]     # Joint velocities  (29,)

        # PD controller: τ = Kp(q_target - q) - Kd * q̇
        torque = KP * (target_q - q) - KD * qd
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

        # Step the physics simulation
        self.do_simulation(torque, self.frame_skip)

        # ── Observation ───────────────────────────────────────────────────────
        obs = self._get_obs()

        # ── Reward ────────────────────────────────────────────────────────────
        pelvis_height = self.data.qpos[2]
        pelvis_xmat = self.data.body("pelvis").xmat.reshape(3, 3)
        upright_value = float(pelvis_xmat[2, 2])  # 1.0 when perfectly upright

        # 1. Alive bonus — dominant signal to encourage survival
        alive_bonus = REWARD_ALIVE

        # 2. Height reward — Gaussian peaked at standing height
        height_error = pelvis_height - STANDING_HEIGHT
        height_reward = REWARD_HEIGHT * np.exp(
            -HEIGHT_GAUSSIAN_K * height_error ** 2
        )

        # 3. Upright reward — dot product of pelvis z-axis with world z
        upright_reward = REWARD_UPRIGHT * max(0.0, upright_value)

        # 4. Energy penalty — discourages wasteful joint actuation
        energy_penalty = PENALTY_ENERGY * float(np.sum(np.abs(torque * qd)))

        # 5. Joint velocity penalty — encourages stillness
        vel_penalty = PENALTY_JOINT_VEL * float(np.sum(qd ** 2))

        # 6. Action magnitude penalty — encourages small corrective actions
        action_penalty = PENALTY_ACTION * float(np.sum(action ** 2))

        # 7. Posture penalty — stay near the natural standing pose
        posture_error = q - STANDING_POSE
        posture_penalty = PENALTY_POSTURE * float(np.sum(posture_error ** 2))

        # 8. COM drift penalty — prevents slow horizontal drift from origin
        #    This is the key fix for "falling backwards after 1000+ steps":
        #    the robot's center of mass must stay centered over its feet.
        pelvis_x = self.data.qpos[0]
        pelvis_y = self.data.qpos[1]
        com_drift_penalty = PENALTY_COM_DRIFT * float(
            pelvis_x ** 2 + pelvis_y ** 2
        )

        # 9. Base angular velocity penalty — prevents rotational drift
        #    accumulation that slowly tips the robot backward.
        base_angvel = self.data.qvel[3:6]
        base_angvel_penalty = PENALTY_BASE_ANGVEL * float(
            np.sum(base_angvel ** 2)
        )

        reward = float(
            alive_bonus
            + height_reward
            + upright_reward
            + energy_penalty
            + vel_penalty
            + action_penalty
            + posture_penalty
            + com_drift_penalty
            + base_angvel_penalty
        )

        # ── Termination ──────────────────────────────────────────────────────
        terminated = bool(
            pelvis_height < 0.4         # Fell too low
            or upright_value < 0.75     # Tilted beyond ~40 deg
        )

        # ── Truncation ───────────────────────────────────────────────────────
        truncated = bool(self._step_count >= MAX_EPISODE_STEPS)

        # ── Info ──────────────────────────────────────────────────────────────
        info = {
            "height": pelvis_height,
            "upright": upright_value,
            "episode_step": self._step_count,
            "pelvis_xy": (pelvis_x, pelvis_y),
            "reward_alive": alive_bonus,
            "reward_height": float(height_reward),
            "reward_upright": float(upright_reward),
            "penalty_energy": float(energy_penalty),
            "penalty_velocity": float(vel_penalty),
            "penalty_action": float(action_penalty),
            "penalty_posture": float(posture_penalty),
            "penalty_com_drift": float(com_drift_penalty),
            "penalty_base_angvel": float(base_angvel_penalty),
        }

        return obs, reward, terminated, truncated, info


# ─── Factory for Parallel Environments ────────────────────────────────────────

def make_env(rank: int, seed: int = 0):
    """
    Factory function for creating parallel environments.

    Parameters
    ----------
    rank : int
        Environment index (for seeding).
    seed : int
        Base random seed.

    Returns
    -------
    callable
        A function that creates and returns a G1StandEnv instance.
    """
    def _init():
        env = G1StandEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init
