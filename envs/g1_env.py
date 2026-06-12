from g1_config import REWARD_ALIVE
import os
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from g1_config import *

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
        qpos += self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        
        qvel = np.zeros(self.model.nv) 
        qvel[6:] = self.np_random.uniform(-0.01, 0.01, size=self.model.nv)

        self.set_state(qpos, qvel)

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

        # if(self._step_count == 1000) :
        #     reward += 100

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