import types
import mujoco
import numpy as np
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.g1_env import G1Env

platform_pos = np.array([0.0, 0.0, -0.05])
platform_vel = np.array([0.1, 0.0, 0.0])
platform_acc = np.array([0.05, 0.0, 0.0])

def custom_do_simulation(self, ctrl, n_frames):
    global platform_pos, platform_vel
    self.data.ctrl[:] = ctrl
    for _ in range(n_frames):
        platform_vel += platform_acc * self.model.opt.timestep
        platform_pos += platform_vel * self.model.opt.timestep
        self.data.mocap_pos[0] = platform_pos

        mujoco.mj_step(self.model, self.data)

env = DummyVecEnv([lambda: G1Env(render_mode="human")])
env = VecNormalize.load("models/g1_stand_force_vecnorm_2.pkl", env)
env.training = False
env.norm_reward = False
env.envs[0].do_simulation = types.MethodType(custom_do_simulation, env.envs[0])
model = PPO.load("models/g1_stand_force_2")
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