from time import sleep
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv #MujocoEnv(wrapper) internally handles the MuJoCo import
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

MODEL_PATH = "C:/Users/admin/Desktop/rl-humanoid/double_pendulum_cartpole.xml"
# MODEL_PATH = os.path.join(os.path.dirname(__file__),"double_pendulum_cartpole.xml")   #Universal

class MyCartPoleEnv(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

        super().__init__(model_path=MODEL_PATH, frame_skip=1, observation_space=observation_space, render_mode=render_mode)

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):

        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset_model(self):

        self.set_state(qpos=np.array([0.0, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)]), qvel=np.array([0.0, 0.0, 0.0]))
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        cart_pos = obs[0]
        pole_angle = obs[1]
        pole_angle_2 = obs[2]
        reward = 1.0 - 0.5 * pole_angle**2 - 0.5 * pole_angle_2**2 - 0.01 * cart_pos**2         #Used AI Help here
        terminated = bool(abs(cart_pos) > 2.0 or abs(pole_angle) > 0.5 or abs(pole_angle_2) > 0.5)
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)


def make_env(rank: int):
    def _init():
        env = MyCartPoleEnv(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init

if __name__ == "__main__":      #Windows Guard(only needed in Windows)
    num_env = 8
    train_env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    # train_env = MyCartPoleEnv(render_mode=None)       # For Single Training
    check_env(MyCartPoleEnv())                          # Check Env (one time only)
    print("Env Check SuccessFul")

    model = PPO(policy="MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, verbose=0, tensorboard_log = "./tb_logs/")

    # Load and continue training
    # model = PPO.load("ppo_cartpole", env=train_env)
    # model.learn(total_timesteps=50_000, reset_num_timesteps=False)
    # model.save("ppo_cartpole_v2")

    print("\nStarting PPO training...")
    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save("ppo_cartpole")
    train_env.close()

    env = MyCartPoleEnv(render_mode="human")
    obs, info = env.reset()
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if step == 500:                                            # Test by giving jerk
            env.data.qvel[0] += 0.3
        env.render()
        # print(f"Step: {step} Observation: {obs} Reward: {reward}")
        sleep(0.02)
        if terminated or truncated:
            print("Episode ended at step {step}. Resetting...\n")
            obs, info = env.reset()
    env.close()