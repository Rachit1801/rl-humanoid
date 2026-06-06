from time import sleep
import mujoco.viewer
from stable_baselines3 import PPO
from envs.cartpole_env import MyCartPoleEnv

print("Loading Model...")
env = MyCartPoleEnv(render_mode="None")
model = PPO.load("models/training_data")
obs, info = env.reset()
viewer = mujoco.viewer.launch_passive(env.model,env.data)

while viewer.is_running():
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    viewer.sync()
    # print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(0.02)
    if terminated or truncated:
        print("Episode ended at step {step}. Resetting...\n")
        obs, info = env.reset()
env.close()