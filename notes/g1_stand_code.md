# Files

## g1_config.py

```python
import numpy as np 
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__),"..","assets","scene_29dof.xml")

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

# Max per-step reward ~ 7.0 (standing perfectly still at origin)
REWARD_ALIVE         =  2.0
REWARD_HEIGHT        =  3.0
REWARD_UPRIGHT       =  2.0     # Alignment of pelvis z-axis with world z
PENALTY_ENERGY       = -0.001   # |torque x joint_vel|
PENALTY_JOINT_VEL    = -0.001   # joint_vel^2  (doubled: punish drift harder)
PENALTY_ACTION       = -0.01    # action^2
PENALTY_POSTURE      = -0.15    # (joint_pos - standing_pose)^2  (increased)
PENALTY_COM_DRIFT    = -3.0     # xy position drift from origin (NEW)
PENALTY_BASE_ANGVEL  = -0.1     # base angular velocity^2 (prevents rotational drift)
HEIGHT_GAUSSIAN_K   = 40.0  

```

## g1_env.py

```python
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from envs.g1_config import *

class G1Env(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float64)

        super().__init__(model_path=MODEL_PATH, frame_skip=5, observation_space=observation_space, render_mode=render_mode)

        self.action_space = Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

        self._step_count = 0

    def _get_obs(self):

        pelvis_xmat = self.data.body("pelvis").xmat.reshape(3, 3)
        base_ang_vel = self.data.qvel[3:6]
        body_ang_vel = pelvis_xmat.T @ base_ang_vel
        base_lin_vel = self.data.qvel[0:3]
        body_lin_vel = pelvis_xmat.T @ base_lin_vel 
        projected_gravity = pelvis_xmat.T @ np.array([0.0, 0.0, -1.0])
        joint_pos = self.data.qpos[7:] - STANDING_POSE
        joint_vel = self.data.qvel[6:]
        
        return np.concatenate([
            body_ang_vel,body_lin_vel,projected_gravity,joint_pos,joint_vel
        ], dtype=np.float64)        # 67
       
    def reset_model(self):

        qpos = np.zeros(self.model.nq)
        qpos[2] = STANDING_HEIGHT       # z
        qpos[3] = 1.0                   # quaternion w
        qpos[7:] = STANDING_POSE.copy()
        qpos[7:] += self.np_random.uniform(-0.02, 0.02, size=29)
        
        qvel = np.zeros(self.model.nv) 
        qvel[6:] = self.np_random.uniform(-0.01, 0.01, size=29)

        self.set_state(qpos, qvel)
        self._step_count = 0
        return self._get_obs()

    def step(self, action):

        self._step_count += 1

        target_q = STANDING_POSE + ACTION_SCALE * action

        q = self.data.qpos[7:]      # joint positions
        qd = self.data.qvel[6:]     # joint velocities

        torque = kp * (target_q - q) - kd * qd
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

        self.do_simulation(torque, self.frame_skip)

        obs = self._get_obs()
        
        height = self.data.qpos[2]
        height_reward = REWARD_HEIGHT * np.exp(-HEIGHT_GAUSSIAN_K * (height - STANDING_HEIGHT) ** 2) #Gaussian

        upright  = float(self.data.body("pelvis").xmat[8])   # 1.0 = upright, 0.0 = 90°
        upright_reward = REWARD_UPRIGHT * max(0.0, upright)

        energy = PENALTY_ENERGY * float(np.sum(np.abs(torque * qd)))
        vel_penalty = PENALTY_JOINT_VEL * float(np.sum(qd ** 2))
        #hip_penalty = float(np.sum(np.square(q[[1, 2, 7, 8]])))
        action_penalty = PENALTY_ACTION * float(np.sum(action ** 2))
        posture_penalty = PENALTY_POSTURE * float(np.sum((q - STANDING_POSE) ** 2))
        com_drift_penalty = PENALTY_COM_DRIFT * float(self.data.qpos[0] ** 2 + self.data.qpos[1] ** 2)
        base_angvel_penalty = PENALTY_BASE_ANGVEL * float(np.sum(self.data.qvel[3:6] ** 2))

        reward = REWARD_ALIVE + height_reward + upright_reward + energy + vel_penalty + action_penalty + posture_penalty + com_drift_penalty + base_angvel_penalty

        terminated = bool(height < 0.4 or upright < 0.75)
        truncated = bool(self._step_count >= MAX_EPISODE_STEPS)
        info = {}
        
        return (obs, reward, terminated, truncated, info)

def make_env(rank: int):
    def _init():
        env = G1Env(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init
```

## train.py

```python
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from envs.g1_env import G1Env
from envs.g1_env import make_env

num_env = 8

if __name__ == "__main__":      #Windows Guard(only needed in Windows)
    check_env(G1Env(), warn=True)                          # Check Env (one time only)
    print("Env Check SuccessFul")
    train_env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    # train_env = MyCartPoleEnv(render_mode=None)       # For Single Training
    train_env = VecMonitor(train_env)   # tracks episode rewards & lengths
    train_env = VecNormalize(train_env,norm_obs=True,norm_reward=True,clip_obs=10.0,clip_reward=10.0,gamma=0.99,)

    model = PPO(policy="MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log = "./tb_logs/")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // 8, 1),
        save_path=os.path.join("models", "checkpoints"),
        name_prefix="g1_stand_retry",
        save_vecnormalize=True,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback])

    print("\nStarting PPO training...")
    model.learn(total_timesteps=3_000_000,callback=callbacks, progress_bar=True)
    model.save("models/g1_stand_retry")
    train_env.save("models/g1_stand_retry_vecnorm.pkl")
    print("\nTraining Complete")
    train_env.close()


    # ── Resume Training (uncomment to continue from checkpoint) ───────────────
    # """
    # checkpoint = "models/checkpoints/g1_stand_XXXXX_steps"
    # vecnorm_checkpoint = "models/checkpoints/g1_stand_XXXXX_steps_vecnormalize.pkl"
    #
    # train_env = SubprocVecEnv([make_env(i, seed=42) for i in range(NUM_ENVS)])
    # train_env = VecMonitor(train_env)
    # train_env = VecNormalize.load(vecnorm_checkpoint, train_env)
    #
    # model = PPO.load(checkpoint, env=train_env)
    # model.learn(total_timesteps=1_000_000, reset_num_timesteps=False,
    #             callback=callbacks, progress_bar=True, tb_log_name="g1_stand")
    # model.save("models/g1_stand_v2")
    # train_env.save("models/g1_stand_v2_vecnorm.pkl")
    # train_env.close()
    # """
```

## run.py

```python
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.g1_env import G1Env

# env = G1Env(render_mode="human")
env = DummyVecEnv([lambda: G1Env(render_mode="human")])
env = VecNormalize.load("models/g1_stand_retry_vecnorm.pkl", env)
model = PPO.load("models/g1_stand_retry")
obs = env.reset()
for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action) 
    env.render()
    # print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(0.02)
    if done[0]:
        print("Episode ended at step {step}. Resetting...\n")
        obs = env.reset()
env.close()
```

---

# Code Explaination

## VecMonitor

Without VecMonitor robot stands, robot falls, PPO trains but SB3 doesn't keep track of episode statistics.

With train_env = VecMonitor(train_env) SB3 automatically records `rollout/ep_len_mean` and `rollout/ep_rew_mean` for TensorBoard. You used these graphs to prove your robot learned standing. It doesn't help the robot learn. It helps you understand training.

## VecNormalize

Suppose observations are `Joint Position  = 0.1` `Joint Velocity  = 20` `Gravity Vector  = -1`. Different scales. Neural networks hate this. VecNormalize computes `normalized = (x - mean) / std` so every observation dimension is roughly mean ≈ 0 and std  ≈ 1 which is ideal for neural networks.

Example:

```
Height:
mean = 0.79
std  = 0.01

height = 0.80

normalized = (0.80 - 0.79)/0.01
           = 1.0
```

### vecnorm.pkl

g1_stand_retry_vecnorm.pkl stores observation mean, observation std, reward mean, reward std learned during training. Without it the policy receives completely different inputs and performance collapses i.e. Training observation is normalized but Running observation is raw

## Checkpoint

Every `save_freq = 50000 // 8` SB3 saves. Suppose 3 million step training takes 50 minutes and PC crashes at 2.8 million steps. TL;DR without checkpoint start again from 0 with checkpoint resume from 2.75M

### Callback

A callback is code that runs automatically during training. Example every 50k steps, save model. That's a callback. 

## DummyVecEnv

environment `G1Env()` is a normal environment. SB3 expects vectorized environments. DummyVecEnv wraps `G1Env()` into `[Env1]` instead of `Env1`. Training uses `SubprocVecEnv` because 8 environments run simultaneously. 

Running uses `DummyVecEnv` because 1 environment only.

## render_fps

```
metadata = {
    "render_fps":100
}
```

Means 100 frames/sec is the intended rendering rate. It only affects visualization timing. It does NOT affect physics.

## observation dimension

```
body_ang_vel       = 3
body_lin_vel       = 3
gravity            = 3
joint_pos          = 29
joint_vel          = 29
```

## xmat

This is MuJoCo. Not Gymnasium. Every body has Position and Orientation. Orientation is stored as 3x3 rotation matrix called `body.xmat`. Example: `pelvis_xmat = self.data.body("pelvis").xmat` returns `[ r11 r12 r13 r21 r22 r23 r31 r32 r33 ]` flattened.

## model.nq

Number of generalized coordinates. Your robot: `7 free-joint coordinates + 29 joints = 36` 

## model.nv

Number of generalized velocities. Free joint velocity `3 linear + 3 angular = 6 + 29 joint velocities = 35 `

---

