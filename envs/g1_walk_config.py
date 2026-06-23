"""
g1_walk_config.py  — Configuration for the G1 walking environment.

Translated from Isaac Lab  velocity_env_cfg.py  (unitree_rl_lab)
to MuJoCo + Gymnasium + Stable-Baselines3.

Joint ordering (29 DoF):
  Left leg:   0 hip_pitch, 1 hip_roll, 2 hip_yaw, 3 knee, 4 ankle_pitch, 5 ankle_roll
  Right leg:  6 hip_pitch, 7 hip_roll, 8 hip_yaw, 9 knee, 10 ankle_pitch, 11 ankle_roll
  Waist:      12 yaw, 13 roll, 14 pitch
  Left arm:   15 shoulder_p, 16 shoulder_r, 17 shoulder_y, 18 elbow, 19 wrist_r, 20 wrist_p, 21 wrist_y
  Right arm:  22 shoulder_p, 23 shoulder_r, 24 shoulder_y, 25 elbow, 26 wrist_r, 27 wrist_p, 28 wrist_y
"""

import numpy as np
import os

# ──────────────────────── Model ────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "platform_29dof.xml")

# ──────────────────────── Physical constants ────────────────────────
STANDING_HEIGHT = 0.793          # pelvis height from XML

TORQUE_LIMITS = np.array([
    88, 88, 88, 139, 50, 50,      # left leg
    88, 88, 88, 139, 50, 50,      # right leg
    88, 50, 50,                    # waist
    25, 25, 25, 25, 25, 5, 5,     # left arm
    25, 25, 25, 25, 25, 5, 5,     # right arm
], dtype=np.float64)

# PD gains — from unitree_rl_lab deployment config
kp = np.array([
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    200, 200, 200,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
], dtype=np.float64)

kd = np.array([
    2, 2, 2, 4, 2, 2,
    2, 2, 2, 4, 2, 2,
    5, 5, 5,
    10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10,
], dtype=np.float64)

# ──────────────────────── Standing pose ────────────────────────
# Natural slightly-bent pose for walking.  The default-offset that
# Isaac Lab applies via `use_default_offset=True` in the G1 URDF.
STANDING_POSE = np.array([
    -0.1,  0,    0,   0.3, -0.2,  0,       # left leg
    -0.1,  0,    0,   0.3, -0.2,  0,       # right leg
     0,    0,    0,                          # waist
     0,    0.25, 0,   0.97, 0.15, 0, 0,    # left arm
     0,   -0.25, 0,   0.97,-0.15, 0, 0,    # right arm
], dtype=np.float64)

# ──────────────────────── Action scaling ────────────────────────
# Per-joint scale  (action ∈ [-1,1] → target = STANDING_POSE + scale*action)
ACTION_SCALE = np.array([
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,   # left leg  (knee gets extra range)
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,   # right leg
    0.15, 0.15, 0.15,                       # waist
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,  # left arm
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,  # right arm
], dtype=np.float64)

# ──────────────────────── Episode ────────────────────────
MAX_EPISODE_STEPS = 2000         # 20 s  (dt=0.002 × frame_skip=5 → 0.01 s/step)

# ──────────────────────── Observation ────────────────────────
# per-frame: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3) + jpos(29) + jvel(29) + last_act(29) = 99
OBS_PER_FRAME  = 99
HISTORY_LENGTH = 5               # Isaac Lab:  history_length = 5
OBS_DIM        = OBS_PER_FRAME * HISTORY_LENGTH   # 495

# Observation scales  (applied after noise → func → noise → scale)
SCALE_LIN_VEL   = 2.0            # Isaac Lab: base_lin_vel scale=2.0
SCALE_ANG_VEL   = 0.2            # Isaac Lab: base_ang_vel  scale=0.2
SCALE_JOINT_VEL = 0.05           # Isaac Lab: joint_vel_rel scale=0.05

# Observation noise  (additive uniform, applied to RAW value BEFORE scaling)
NOISE_LIN_VEL   = 0.1            # ±0.1 m/s
NOISE_ANG_VEL   = 0.2            # ±0.2 rad/s
NOISE_GRAVITY   = 0.05           # ±0.05
NOISE_JOINT_POS = 0.01           # ±0.01 rad
NOISE_JOINT_VEL = 1.5            # ±1.5 rad/s

# ──────────────────────── Velocity commands ────────────────────────
# Initial (easy) and limit (hard) ranges — curriculum interpolates between them
CMD_VX_INIT  = (-0.1,  0.1)
CMD_VX_LIMIT = (-0.3,  0.3)      # reduced from Isaac Lab (−0.5, 1.0) to fit on 2m platform
CMD_VY_INIT  = (-0.1,  0.1)
CMD_VY_LIMIT = (-0.2,  0.2)      # reduced from Isaac Lab (−0.3, 0.3)
CMD_WZ_INIT  = (-0.1,  0.1)
CMD_WZ_LIMIT = (-0.2,  0.2)

CMD_RESAMPLE_TIME   = 10.0       # seconds between velocity-command resamples
STANDING_PROBABILITY = 0.02      # 2% of resamples → zero cmd  (Isaac Lab: rel_standing_envs)

# ──────────────────────── Reward weights  (from velocity_env_cfg RewardsCfg) ────────────
# task
WEIGHT_TRACK_LIN_VEL     =  1.0
WEIGHT_TRACK_ANG_VEL     =  0.5
WEIGHT_ALIVE             =  0.15
# base
WEIGHT_LIN_VEL_Z         = -2.0
WEIGHT_ANG_VEL_XY        = -0.05
WEIGHT_JOINT_VEL         = -0.001
WEIGHT_JOINT_ACC         = -2.5e-7
WEIGHT_ACTION_RATE       = -0.05
WEIGHT_DOF_POS_LIMITS    = -5.0
WEIGHT_ENERGY            = -2e-5
# joint deviation
WEIGHT_JOINT_DEV_ARMS    = -0.1
WEIGHT_JOINT_DEV_WAIST   = -1.0
WEIGHT_JOINT_DEV_LEGS    = -1.0
# orientation / height
WEIGHT_FLAT_ORIENTATION  = -5.0
WEIGHT_BASE_HEIGHT       = -10.0
TARGET_HEIGHT            =  0.78          # walking height (slightly crouched vs 0.793 standing)
# gait
WEIGHT_GAIT              =  0.5
WEIGHT_FEET_SLIDE        = -0.2
WEIGHT_FEET_CLEARANCE    =  1.0
# contacts
WEIGHT_UNDESIRED_CONTACTS = -1.0
# platform drift  (keeps robot centered on 2m platform)
WEIGHT_COM_DRIFT         = -5.0
# termination
WEIGHT_TERMINATE         = -200.0

# ──────────────────────── Gait parameters ────────────────────────
GAIT_PERIOD      = 0.8           # seconds per stride cycle
GAIT_OFFSETS     = [0.0, 0.5]    # [left, right] — perfect alternation
GAIT_THRESHOLD   = 0.55          # fraction-of-period that counts as stance

FOOT_CLEARANCE_TARGET    = 0.1   # metres (swing-foot target height above platform surface)
FOOT_CLEARANCE_STD       = 0.05
FOOT_CLEARANCE_TANH_MULT = 2.0

# ──────────────────────── Velocity-tracking kernel ────────────────────────
TRACKING_SIGMA_SQ = 0.25         # exp( -error² / sigma² )

# ──────────────────────── Termination ────────────────────────
TERM_MIN_HEIGHT  = 0.2           # metres — fell down
TERM_MAX_TILT    = 0.8           # radians — cos(0.8) ≈ 0.697
TERM_MAX_DRIFT   = 0.85          # metres from platform centre (platform half-size = 1m)

# ──────────────────────── Domain randomisation ────────────────────────
PUSH_INTERVAL_RANGE = (400, 600) # env-steps between pushes  (~4-6 s)
PUSH_VEL_RANGE      = (-0.5, 0.5)  # m/s velocity impulse (Isaac Lab: push_by_setting_velocity)
FRICTION_RANGE       = (0.3, 1.0)   # randomised every reset

# ──────────────────────── Platform ────────────────────────
PLATFORM_ACCEL_MAX  = 0.3        # m/s² max platform velocity ramp
PLATFORM_VEL_LIMIT  = 0.3        # m/s max perturbation velocity at full curriculum

# ──────────────────────── Curriculum ────────────────────────
CURRICULUM_STEP_UP            = 0.05   # level increase after a successful episode
CURRICULUM_STEP_DOWN          = 0.02   # level decrease after a failed episode
CURRICULUM_SURVIVAL_THRESHOLD = 0.8    # survive ≥80% of max steps to count as success
CURRICULUM_TRACKING_THRESHOLD = 0.4    # average tracking reward ≥0.4

# ──────────────────────── Joint-group indices ────────────────────────
# Used for per-group deviation penalties (matching Isaac Lab's SceneEntityCfg joint_names)
ARM_JOINT_INDICES     = list(range(15, 29))     # shoulder_*, elbow, wrist_*
WAIST_JOINT_INDICES   = [12, 13, 14]            # waist yaw, roll, pitch
LEG_DEV_JOINT_INDICES = [1, 2, 7, 8]            # hip_roll + hip_yaw both legs
