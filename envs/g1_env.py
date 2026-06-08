import os
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

MODEL_PATH = os.path.join(os.path.dirname(__file__),"..","assets","scene_29dof.xml")

TORQUE_LIMITS = np.array([
    88, 88, 88, 139, 50, 50,        # Left Leg
    88, 88, 88, 139, 50, 50,        # Right Leg
    88, 50, 50,                     # Waist
    25, 25, 25, 25, 25, 5, 5,       # Left Arm
    25, 25, 25, 25, 25, 5, 5,       # Right Arm
], dtype=np.float64)                # shape (29,)

STANDING_HEIGHT = 0.793             # from XML:  pos="0 0 0.793"


class G1Env(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(69,), dtype=np.float64)

        super().__init__(model_path=MODEL_PATH, frame_skip=5, observation_space=observation_space, render_mode=render_mode)

        self.action_space = Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

    def _get_obs(self):

        return np.concatenate([self.data.qpos[2:], self.data.qvel])     # Skip global x, y as we are learning to stand

    def reset_model(self):

        qpos = np.zeros(self.model.nq)
        qpos[0] = 0.0             # x
        qpos[1] = 0.0             # y
        qpos[2] = STANDING_HEIGHT # z
        qpos[3] = 1.0             # quaternion w
        qpos[4] = 0.0             # quaternion x
        qpos[5] = 0.0             # quaternion y
        qpos[6] = 0.0             # quaternion z
        qpos[7:] = self.np_random.uniform(-0.05, 0.05, size=29)

        qvel = self.np_random.uniform(-0.05, 0.05, size=self.model.nv)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):

        target_q = np.zeros(29)

        target_q[0] = -0.2    # left hip pitch
        target_q[3] =  0.4    # left knee
        target_q[4] = -0.2    # left ankle pitch

        target_q[6] = -0.2    # right hip pitch
        target_q[9] =  0.4    # right knee
        target_q[10]= -0.2    # right ankle pitch

        target_q = target_q + 0.15 * action

        q = self.data.qpos[7:]      # joint positions
        qd = self.data.qvel[6:]     # joint velocities
        kp = 100.0
        kd = 5.0
        torque = kp * (target_q - q) - kd * qd
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.do_simulation(torque, self.frame_skip)

        obs = self._get_obs()
        
        height = self.data.qpos[2]
        height_reward = float(np.clip(height/STANDING_HEIGHT, 0.0, 1.0)) * 2
        upright = float(self.data.body("pelvis").xmat[8]) * 2
        energy = 0.001 * float(np.sum(action**2))

        reward = 0.5 + height_reward + upright - energy

        terminated = bool(height < 0.35 or upright < 0.5)
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)

def make_env(rank: int):
    def _init():
        env = G1Env(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init