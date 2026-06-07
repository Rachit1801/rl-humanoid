from time import sleep
from stable_baselines3 import PPO
from envs.g1_env import G1Env

env = G1Env(render_mode="human")
model = PPO.load("models/g1_stand")
obs, info = env.reset()
for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action) 
    env.render()
    # print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(0.02)
    if terminated or truncated:
        print("Episode ended at step {step}. Resetting...\n")
        obs, info = env.reset()
env.close()