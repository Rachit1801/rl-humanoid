"""Quick validation of the updated G1 stand environment."""
from stable_baselines3.common.env_checker import check_env
from envs.g1_stand_env import G1StandEnv
import numpy as np

print("Creating G1StandEnv (v2 - with drift penalties)...")
env = G1StandEnv()

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

print("\nRunning check_env...")
check_env(env, warn=True)
print("check_env PASSED")

print("\nRunning 20-step rollout with random actions...")
obs, info = env.reset()

total_reward = 0
for i in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if i == 0:
        print(f"  Step 1: reward={reward:.3f}, height={info['height']:.3f}, upright={info['upright']:.3f}")
        print(f"          com_drift={info['penalty_com_drift']:.4f}, base_angvel={info['penalty_base_angvel']:.4f}")

print(f"\n  20-step total reward: {total_reward:.3f}")
print(f"  Final height: {info['height']:.3f}")
print(f"  Final pelvis_xy: ({info['pelvis_xy'][0]:.4f}, {info['pelvis_xy'][1]:.4f})")

env.close()
print("\nALL CHECKS PASSED")
