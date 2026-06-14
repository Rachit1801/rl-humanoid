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