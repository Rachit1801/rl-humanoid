from time import sleep
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv #MujocoEnv(wrapper) internally handles the MuJoCo import
from gymnasium.spaces import Box
import numpy as np

class MyCartPoleEnv(MujocoEnv):

    def __init__(self):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        super().__init__(model_path="C:/Users/admin/Desktop/rl-humanoid/cartpole.xml", frame_skip=1, observation_space=observation_space, render_mode="human")

    def _get_obs(self):

        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset_model(self):

        self.set_state(qpos=np.array([0.0, 0.05]), qvel=np.array([0.0, 0.0]))
        return self._get_obs()

    def step(self, action):

        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        cart_pos = obs[0]
        pole_angle = obs[1]
        reward = 1.0
        terminated = bool(abs(cart_pos) > 2.0 or abs(pole_angle) > 0.5)
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)

env = MyCartPoleEnv()
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(0.02)
    if terminated or truncated:
        print("Episode ended. Resetting...\n")
        obs, info = env.reset()
env.close()