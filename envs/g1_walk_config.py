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
# Physical parameters  (identical to g1_config.py)
# ═══════════════════════════════════════════════════════════════════════════════
TORQUE_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,        # left leg   (7520_14, 7520_22, 7520_14, 7520_22, 2×5020, 2×5020)
    88, 139, 88, 139, 50, 50,        # right leg
    88, 50, 50,                      # waist      (7520_14, 2×5020, 2×5020)
    25, 25, 25, 25, 25, 5, 5,        # left arm   (5020 ×5, 4010 ×2)
    25, 25, 25, 25, 25, 5, 5,        # right arm
], dtype=np.float64)

# Action scale: 0.25 × effort_limit / stiffness  (from official g1_constants.py)
ACTION_SCALE = np.array([
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44,             # left leg
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44,             # right leg
    0.55, 0.44, 0.44,                                # waist
    0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,       # left arm
    0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,       # right arm
], dtype=np.float64)

# PD gains derived from official unitree_rl_mjlab actuator model (g1_constants.py):
#   ω = 10 × 2π (natural freq 10 Hz), ζ = 2.0 (damping ratio)
#   kp = armature × ω²,  kd = 2 × ζ × armature × ω
#
# Actuator → joints:
#   7520_14 (88 N·m) : hip_pitch, hip_yaw, waist_yaw
#   7520_22 (139 N·m): hip_roll, knee
#   2×5020  (50 N·m) : ankle_pitch, ankle_roll, waist_roll, waist_pitch
#   5020    (25 N·m) : shoulder_p/r/y, elbow, wrist_roll
#   4010    (5 N·m)  : wrist_pitch, wrist_yaw
kp = np.array([
    40.19, 99.09, 40.19, 99.09, 28.50, 28.50,   # left leg
    40.19, 99.09, 40.19, 99.09, 28.50, 28.50,   # right leg
    40.19, 28.50, 28.50,                         # waist
    14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78,  # left arm
    14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78,  # right arm
], dtype=np.float64)

kd = np.array([
    2.56, 6.31, 2.56, 6.31, 1.82, 1.82,         # left leg
    2.56, 6.31, 2.56, 6.31, 1.82, 1.82,         # right leg
    2.56, 1.82, 1.82,                            # waist
    0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07,   # left arm
    0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07,   # right arm
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
#   body_ang_vel  (3)  +  body_lin_vel (3)  +  projected_gravity (3)  +
#   vel_command   (3)  +  phase        (2)  +  joint_pos         (29) +
#   joint_vel    (29)  +  last_action (29)  +  platform_vel       (2)
#   ─────────────────────────────────────────────────────────────────
#   Total = 103
# ═══════════════════════════════════════════════════════════════════════════════
OBS_DIM = 103

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

# ═══════════════════════════════════════════════════════════════════════════════
# Reward weights
#
# Positive rewards — behaviours we want
# Negative penalties — behaviours we discourage
#
# Proportions follow the official velocity_env_cfg.py reward table.
# The alive / height / upright terms are kept from the user's existing standing
# env because they provide a strong learning signal for SB3 PPO; the official
# RSL-RL pipeline doesn't need them due to different reward normalisation.
# ═══════════════════════════════════════════════════════════════════════════════
REWARD_ALIVE         =  1.0       # per-step survival bonus
REWARD_HEIGHT        =  2.0       # Gaussian  exp(-K*(h-h0)^2)
REWARD_UPRIGHT       =  1.5       # pelvis z-axis alignment with world z
REWARD_VEL_TRACKING  =  3.0       # exp velocity tracking  (official weight 1.0, boosted for SB3)
REWARD_GAIT          =  1.5       # phase-matched foot contacts  (official 0.5)
REWARD_POSTURE       =  1.0       # exp variable posture  (official 1.0)

PENALTY_ENERGY       = -0.001     # |torque × joint_vel|
PENALTY_JOINT_VEL    = -0.001     # joint_vel²
PENALTY_ACTION       = -0.005     # action²
PENALTY_ACTION_RATE  = -0.05      # (a − a_prev)²   (official −0.05)
PENALTY_COM_DRIFT    = -0.5       # xy drift from platform centre
PENALTY_BASE_ANGVEL  = -0.05      # base angular velocity²  (official −0.05)
PENALTY_FOOT_SLIP    = -0.25      # foot xy vel² when in contact  (official −0.25)
PENALTY_TERMINATION  = -50.0      # one-time cost for falling

REWARD_STAND_STILL   = -1.0       # joint deviation² when standing  (official −1.0)

HEIGHT_GAUSSIAN_K    = 40.0       # sharpness of height Gaussian

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
# Curriculum stages
#
# Each env independently tracks its own level and promotes/demotes at the
# end of every episode based on performance, similar to the official
# terrain_levels_vel() in curriculums.py which moves robots to harder/easier
# terrain blocks based on how far they walked.
#
# Here we ramp three axes:
#   1. Forward velocity command range
#   2. Platform disturbance magnitude
#   3. Episode length
# ═══════════════════════════════════════════════════════════════════════════════
CURRICULUM_STAGES = {
    0: {  # Stand & balance on gently moving platform
        "vel_x_range":        (0.0, 0.0),
        "platform_vel_max":   0.1,
        "platform_accel_max": 0.1,
        "max_episode_steps":  1000,       # 10 s
    },
    1: {  # Slow forward walking
        "vel_x_range":        (0.0, 0.3),
        "platform_vel_max":   0.2,
        "platform_accel_max": 0.15,
        "max_episode_steps":  1500,       # 15 s
    },
    2: {  # Moderate walking
        "vel_x_range":        (0.0, 0.6),
        "platform_vel_max":   0.3,
        "platform_accel_max": 0.2,
        "max_episode_steps":  2000,       # 20 s
    },
    3: {  # Walking with stronger disturbances
        "vel_x_range":        (0.0, 0.8),
        "platform_vel_max":   0.5,
        "platform_accel_max": 0.3,
        "max_episode_steps":  2000,
    },
    4: {  # Full difficulty
        "vel_x_range":        (0.0, 1.0),
        "platform_vel_max":   0.8,
        "platform_accel_max": 0.5,
        "max_episode_steps":  2000,
    },
}

NUM_CURRICULUM_STAGES = len(CURRICULUM_STAGES)

# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive curriculum — promote / demote thresholds
#
# Inspired by the official terrain_levels_vel() which promotes a robot if it
# walked > half the terrain block, and demotes if it walked < 50 % of what the
# velocity command asked.
#
# Here we use survival fraction (episode steps / max episode steps).
#   - PROMOTE if survived > 80 % of the episode AND accumulated good reward
#   - DEMOTE  if survived < 30 % of the episode (fell early)
#
# Each env tracks its own level independently.
#
# consecutive_successes / consecutive_failures prevent noisy single-episode
# flukes from bouncing the level up and down.
# ═══════════════════════════════════════════════════════════════════════════════
PROMOTE_SURVIVAL_FRAC  = 0.80    # must survive ≥ 80 % of episode to be promotion-eligible
DEMOTE_SURVIVAL_FRAC   = 0.30    # survived < 30 % → demotion-eligible
PROMOTE_STREAK         = 3       # need 3 consecutive good episodes to promote
DEMOTE_STREAK          = 2       # need 2 consecutive bad episodes to demote

TOTAL_TIMESTEPS = 20_000_000

