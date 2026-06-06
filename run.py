from time import sleep
from stable_baselines3 import PPO
from envs.cartpole_env import MyCartPoleEnv

env = MyCartPoleEnv(render_mode="human")
model = PPO.load("models/training_data")
obs, info = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if step == 500:                                            # Test by giving jerk
        env.data.qvel[0] += 1
    env.render()
    # print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(0.02)
    if terminated or truncated:
        print("Episode ended at step {step}. Resetting...\n")
        obs, info = env.reset()
env.close()