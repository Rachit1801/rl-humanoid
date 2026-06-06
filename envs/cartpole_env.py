import os
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

MODEL_PATH = os.path.join(os.path.dirname(__file__),"..","assets","cartpole.xml")

#MODEL_PATH = "C:/Users/admin/Desktop/rl-humanoid/double_pendulum_cartpole.xml"

class MyCartPoleEnv(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        super().__init__(model_path=MODEL_PATH, frame_skip=1, observation_space=observation_space, render_mode=render_mode)

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):

        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset_model(self):

        self.set_state(qpos=np.array([0.0, np.random.uniform(-0.05, 0.05)]), qvel=np.array([0.0, 0.0]))
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        cart_pos = obs[0]
        pole_angle = obs[1]
        reward = 1.0
        terminated = bool(abs(cart_pos) > 2.0 or abs(pole_angle) > 0.5)
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)

def make_env(rank: int):
    def _init():
        env = MyCartPoleEnv(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init