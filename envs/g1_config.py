import numpy as np 
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__),"..","assets","platform_29dof.xml")

"""
Variable Structure
Left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Waist:     yaw, roll, pitch
Left arm:  shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
Right arm: shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y
"""
TORQUE_LIMITS = np.array([
    88, 88, 88, 139, 50, 50,        
    88, 88, 88, 139, 50, 50,        
    88, 50, 50,                     
    25, 25, 25, 25, 25, 5, 5,       
    25, 25, 25, 25, 25, 5, 5,       
], dtype=np.float64)

STANDING_HEIGHT = 0.793             # from XML:  pos="0 0 0.793"

ACTION_SCALE = np.array([
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,
    0.15, 0.15, 0.15,
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,
], dtype=np.float64)

# STANDING_POSE = np.array([
#     -0.1, 0, 0, 0.3, -0.2, 0,
#     -0.1, 0, 0, 0.3, -0.2, 0,
#     0, 0, 0,
#     0, 0.25, 0, 0.97, 0.15, 0, 0,
#     0, -0.25, 0, 0.97, -0.15, 0, 0
# ])

STANDING_POSE = np.zeros(29, dtype=np.float64)

kp = np.array([             # kp and kd values taken from unitreerobotics/unitree_rl_lab/deploy/robots/g1_29dof/config/config.yaml
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        200, 200, 200,
        40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40
])

kd = np.array([
        2, 2, 2, 4, 2, 2,
        2, 2, 2, 4, 2, 2,
        5, 5, 5,
        10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10
])

MAX_EPISODE_STEPS = 2000      # Doubled: policy must learn long-term stability

# Rewards weights
REWARD_ALIVE         =  2.0
REWARD_HEIGHT        =  3.0
REWARD_UPRIGHT       =  2.0     # Alignment of pelvis z-axis with world z
PENALTY_ENERGY       = -0.001   # |torque x joint_vel|
PENALTY_JOINT_VEL    = -0.001   # joint_vel^2  (doubled: punish drift harder)
PENALTY_ACTION       = -0.005    # action^2
PENALTY_POSTURE      = -0.15    # (joint_pos - standing_pose)^2  (increased)
PENALTY_COM_DRIFT    = -1.0     # xy position drift from origin (NEW)
PENALTY_BASE_ANGVEL  = -0.1     # base angular velocity^2 (prevents rotational drift)
HEIGHT_GAUSSIAN_K   = 40.0  

PLATFORM_ACCEL_MAX = 0.3