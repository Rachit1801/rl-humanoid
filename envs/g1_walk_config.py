"""
Walking configuration for Unitree G1 with curriculum learning on a moving platform.

Standing pose and height derived from official unitree_rl_mjlab HOME_KEYFRAME.
Reward structure follows official velocity_env_cfg.py and rewards.py.
Curriculum stages progressively increase velocity commands and platform disturbance.

Joint order (29 DOF):
  Left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
  Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
  Waist:     yaw, roll, pitch
  Left arm:  shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
  Right arm: shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
"""

import numpy as np
import os

# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "platform_29dof.xml")

# ═══════════════════════════════════════════════════════════════════════════════
# Physical parameters  (from official g1_constants.py)
# ═══════════════════════════════════════════════════════════════════════════════
TORQUE_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,        # left leg   (7520_14, 7520_22, 7520_14, 7520_22, 2×5020, 2×5020)
    88, 139, 88, 139, 50, 50,        # right leg
    88, 50, 50,                      # waist      (7520_14, 2×5020, 2×5020)
    25, 25, 25, 25, 25, 5, 5,        # left arm   (5020 ×5, 4010 ×2)
    25, 25, 25, 25, 25, 5, 5,        # right arm
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# PD gains — recomputed from official g1_constants.py reflected inertia formula
#
# Each actuator type has a reflected_inertia computed from rotor inertia + gears.
#   ω = 10 × 2π (natural freq 10 Hz), ζ = 2.0 (damping ratio)
#   kp = armature × ω²,  kd = 2 × ζ × armature × ω
#
# Actuator → joints:
#   7520_14 (88 N·m, armature=0.012574) : hip_pitch, hip_yaw, waist_yaw
#   7520_22 (139 N·m, armature=0.032119): hip_roll, knee
#   2×5020  (50 N·m, armature=0.008672) : ankle_pitch, ankle_roll, waist_roll, waist_pitch
#   5020    (25 N·m, armature=0.004336) : shoulder_p/r/y, elbow, wrist_roll
#   4010    (5 N·m, armature=0.004250)  : wrist_pitch, wrist_yaw
# ═══════════════════════════════════════════════════════════════════════════════
kp = np.array([
    49.64, 126.80, 49.64, 126.80, 34.23, 34.23,   # left leg
    49.64, 126.80, 49.64, 126.80, 34.23, 34.23,   # right leg
    49.64, 34.23, 34.23,                            # waist
    17.12, 17.12, 17.12, 17.12, 17.12, 16.78, 16.78,  # left arm
    17.12, 17.12, 17.12, 17.12, 17.12, 16.78, 16.78,  # right arm
], dtype=np.float64)

kd = np.array([
    3.16, 8.07, 3.16, 8.07, 2.18, 2.18,           # left leg
    3.16, 8.07, 3.16, 8.07, 2.18, 2.18,           # right leg
    3.16, 2.18, 2.18,                               # waist
    1.09, 1.09, 1.09, 1.09, 1.09, 1.07, 1.07,     # left arm
    1.09, 1.09, 1.09, 1.09, 1.09, 1.07, 1.07,     # right arm
], dtype=np.float64)

# Action scale: 0.25 × effort_limit / stiffness  (from official g1_constants.py)
ACTION_SCALE = np.array([
    0.4432, 0.2741, 0.4432, 0.2741, 0.3652, 0.3652,   # left leg
    0.4432, 0.2741, 0.4432, 0.2741, 0.3652, 0.3652,   # right leg
    0.4432, 0.3652, 0.3652,                              # waist
    0.3651, 0.3651, 0.3651, 0.3651, 0.3651, 0.0745, 0.0745,  # left arm
    0.3651, 0.3651, 0.3651, 0.3651, 0.3651, 0.0745, 0.0745,  # right arm
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# Standing pose — from official g1_constants.py  HOME_KEYFRAME
#
# Slightly bent knees for a natural walking-ready stance.
# Height lowered from XML default (0.793) → 0.78 so feet press firmly onto the
# platform at initialisation, avoiding the "air-drop" problem observed with bent
# legs at higher spawn heights.
# ═══════════════════════════════════════════════════════════════════════════════
STANDING_HEIGHT = 0.78

STANDING_POSE = np.array([
    # Left leg:  hip_p   hip_r  hip_y  knee   ankle_p  ankle_r
                -0.1,    0.0,   0.0,   0.3,  -0.2,     0.0,
    # Right leg: hip_p   hip_r  hip_y  knee   ankle_p  ankle_r
                -0.1,    0.0,   0.0,   0.3,  -0.2,     0.0,
    # Waist:     yaw     roll   pitch
                 0.0,    0.0,   0.0,
    # Left arm:  sp      sr     sy     elbow  wr       wp      wy
                 0.35,   0.18,  0.0,   0.87,  0.0,     0.0,    0.0,
    # Right arm: sp      sr     sy     elbow  wr       wp      wy
                 0.35,  -0.18,  0.0,   0.87,  0.0,     0.0,    0.0,
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# Observation space
#
# Flat array with actor features first, critic-only features appended at the end.
# The AsymmetricPolicy in train_walk.py zeros out indices ACTOR_OBS_DIM: for actor.
#
#   Actor features (98D):
#     base_ang_vel  (3)  +  projected_gravity (3)  +  vel_command (3)  +
#     phase         (2)  +  joint_pos         (29) +  joint_vel  (29)  +
#     last_action  (29)
#
#   Critic-only features (17D):
#     base_lin_vel  (3)  +  platform_vel      (2)  +  foot_height (2) +
#     foot_air_time (2)  +  foot_contact      (2)  +  foot_contact_forces (6)
#
#   Total = 115
# ═══════════════════════════════════════════════════════════════════════════════
ACTOR_OBS_DIM  = 98
CRITIC_OBS_DIM = 115
OBS_DIM        = CRITIC_OBS_DIM   # env returns full 115D; policy masks for actor

# ═══════════════════════════════════════════════════════════════════════════════
# Gait parameters  (from official velocity_env_cfg.py → foot_gait reward)
# ═══════════════════════════════════════════════════════════════════════════════
GAIT_PERIOD       = 0.6           # seconds per full gait cycle
GAIT_OFFSETS      = [0.0, 0.5]    # left foot phase 0, right foot phase 0.5
GAIT_STANCE_RATIO = 0.56          # fraction of cycle where foot should be in stance

# ═══════════════════════════════════════════════════════════════════════════════
# Velocity command sampling  (from official velocity_command.py)
# ═══════════════════════════════════════════════════════════════════════════════
VEL_CMD_RESAMPLE_TIME = (3.0, 8.0)   # seconds between velocity command resamples
STANDING_CMD_PROB     = 0.05          # probability of zero-velocity command per resample

# ═══════════════════════════════════════════════════════════════════════════════
# Velocity tracking  (from official rewards.py → track_linear_velocity)
# reward = exp( -error / std² )    where  std = sqrt(0.25), std² = 0.25
# ═══════════════════════════════════════════════════════════════════════════════
VEL_TRACKING_SIGMA = 0.25    # = std²; official passes std=sqrt(0.25) and divides by std²
ANG_VEL_TRACKING_SIGMA = 0.5      # std² = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# Reward weights  —  EXACT match to official velocity_env_cfg.py
#
# No alive/height/energy/joint_vel/action rewards — these do not exist in
# the official code and distort the carefully calibrated reward landscape.
# ═══════════════════════════════════════════════════════════════════════════════
# Positive rewards
REWARD_VEL_TRACKING      =  1.0       # exp velocity tracking  (official 1.0)
REWARD_ANG_VEL_TRACKING  =  1.0       # track yaw rate command (official 1.0)
REWARD_GAIT              =  0.5       # phase-matched foot contacts  (official 0.5)
REWARD_POSTURE           =  1.0       # exp variable posture  (official 1.0)

# Negative penalties
PENALTY_BODY_ORIENTATION = -1.0       # projected gravity xy² on torso_link  (official −1.0)
PENALTY_ACTION_RATE      = -0.05      # (a − a_prev)²   (official −0.05)
PENALTY_FOOT_SLIP        = -0.25      # foot xy vel² when in contact  (official −0.25)
PENALTY_JOINT_ACC        = -2.5e-7    # joint_acc²  (official −2.5e-7)
PENALTY_JOINT_POS_LIMITS = -10.0      # penalise nearing joint limits  (official −10.0)
PENALTY_FOOT_CLEARANCE   = -1.0       # wrong foot height during swing  (official −1.0)
PENALTY_SOFT_LANDING     = -1e-3      # impact force penalty  (official −1e-3)
PENALTY_TERMINATION      = -200.0     # one-time cost for falling  (official −200.0)
PENALTY_BASE_ANGVEL      = -0.05      # torso angular velocity²  (official −0.05)
REWARD_STAND_STILL       = -1.0       # joint deviation² when standing  (official −1.0)

# Platform-specific (not in official — needed for our moving platform)
PENALTY_COM_DRIFT        = -0.5       # xy drift from platform centre

FOOT_CLEARANCE_TARGET = 0.10      # official target swing height

# ═══════════════════════════════════════════════════════════════════════════════
# Variable posture standard deviations  (from official env_cfgs.py)
#
# When standing (|vel_cmd| < 0.1) — tight tolerance, robot must hold default pose.
# When walking  (|vel_cmd| ≥ 0.1) — relaxed tolerance, allow natural motion.
# Each element corresponds to one of the 29 joints, in the same order.
# ═══════════════════════════════════════════════════════════════════════════════
STD_STANDING = np.full(29, 0.05, dtype=np.float64)

STD_WALKING = np.array([
    # Left leg:  hip_p  hip_r  hip_y  knee   ankle_p  ankle_r
                 0.5,   0.15,  0.15,  0.5,   0.15,    0.1,
    # Right leg: hip_p  hip_r  hip_y  knee   ankle_p  ankle_r
                 0.5,   0.15,  0.15,  0.5,   0.15,    0.1,
    # Waist:     yaw    roll   pitch
                 0.15,  0.1,   0.1,
    # Left arm:  sp     sr     sy     elbow  wr       wp      wy
                 0.15,  0.1,   0.1,   0.1,   0.1,     0.1,    0.1,
    # Right arm: sp     sr     sy     elbow  wr       wp      wy
                 0.15,  0.1,   0.1,   0.1,   0.1,     0.1,    0.1,
], dtype=np.float64)

STD_RUNNING = np.array([
    # Left leg:  hip_p  hip_r  hip_y  knee   ankle_p  ankle_r
                 0.5,   0.25,  0.25,  0.5,   0.25,    0.1,
    # Right leg: hip_p  hip_r  hip_y  knee   ankle_p  ankle_r
                 0.5,   0.25,  0.25,  0.5,   0.25,    0.1,
    # Waist:     yaw    roll   pitch
                 0.25,  0.1,   0.1,
    # Left arm:  sp     sr     sy     elbow  wr       wp      wy
                 0.25,  0.1,   0.1,   0.1,   0.1,     0.1,    0.1,
    # Right arm: sp     sr     sy     elbow  wr       wp      wy
                 0.25,  0.1,   0.1,   0.1,   0.1,     0.1,    0.1,
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# Variable posture speed thresholds  (from official velocity_env_cfg.py)
# ═══════════════════════════════════════════════════════════════════════════════
WALKING_THRESHOLD = 0.1     # total_speed below this → standing (tight posture)
RUNNING_THRESHOLD = 1.5     # total_speed above this → running  (loose posture)

# ═══════════════════════════════════════════════════════════════════════════════
# Termination
# ═══════════════════════════════════════════════════════════════════════════════
MIN_HEIGHT  = 0.4      # pelvis height below this → terminated
MIN_UPRIGHT = 0.3      # cos(tilt) below this → terminated  (~72°, official uses 70°)

# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum — Step-based velocity expansion
#
# Mirrors the official commands_vel() curriculum in curriculums.py:
#   Stage 0 (steps 0–120k):   moderate velocity range
#   Stage 1 (steps 120k+):    full velocity range
#
# Additionally, we ramp platform disturbance (our addition) alongside velocity.
#
# Note: the official uses "step" = 5000 * 24 = 120,000 env steps for the
# expansion threshold. We keep the same value.
# ═══════════════════════════════════════════════════════════════════════════════
CURRICULUM_VEL_EXPAND_STEP = 120_000   # expand velocity ranges after this many env steps

CURRICULUM_STAGES = {
    0: {  # Initial: moderate velocity, gentle platform
        "vel_x_range":        (-0.5, 1.0),
        "vel_y_range":        (-0.5, 0.5),
        "vel_yaw_range":      (-1.0, 1.0),
        "platform_vel_max":   0.1,
        "platform_accel_max": 0.1,
        "max_episode_steps":  1000,       # 20 s at dt=0.02
    },
    1: {  # Full velocity range, stronger platform
        "vel_x_range":        (-1.0, 2.0),
        "vel_y_range":        (-1.0, 1.0),
        "vel_yaw_range":      (-1.0, 1.0),
        "platform_vel_max":   0.3,
        "platform_accel_max": 0.2,
        "max_episode_steps":  1000,       # 20 s at dt=0.02
    },
}

NUM_CURRICULUM_STAGES = len(CURRICULUM_STAGES)

TOTAL_TIMESTEPS = 20_000_000
