import mujoco
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.g1_env import G1Env

env = DummyVecEnv([lambda: G1Env(render_mode="human")])
env = VecNormalize.load("models/g1_platform_norm_v_2.pkl", env)
env.training = False
env.norm_reward = False
model = PPO.load("models/g1_platfrom_v_2")
obs = env.reset()
for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action) 
    env.render()
    # print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(env.envs[0].dt)
    if done[0]:
        print("Episode ended at step {step}. Resetting...\n")
        obs = env.reset()
    # Cam
    if step == 0:
        viewer = env.envs[0].unwrapped.mujoco_renderer.viewer
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.envs[0].model.body("pelvis").id
env.close()