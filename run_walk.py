"""
Visualise a trained G1 walking policy on the moving platform.

Usage:
    python run_walk.py

If you prefer to adapt your existing run.py instead, the only changes needed are:
  1. Import G1WalkEnv from envs.g1_walk_env  (instead of G1Env from envs.g1_env)
  2. Load the walk model/norm files  (g1_walk_v1 / g1_walk_norm_v1.pkl)
  3. Call env.env_method("set_curriculum_stage", 4)  to set full difficulty
"""

import mujoco
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.g1_walk_env import G1WalkEnv

# ── Environment ─────────────────────────────────────────────────────────
env = DummyVecEnv([lambda: G1WalkEnv(render_mode="human")])
env = VecNormalize.load("models/g1_walk_norm_v1.pkl", env)
env.training = False
env.norm_reward = False

# Set to full difficulty for evaluation
env.env_method("set_curriculum_stage", 0)

# ── Model ───────────────────────────────────────────────────────────────
model = PPO.load("models/g1_walk_v1")

# ── Run ─────────────────────────────────────────────────────────────────
obs = env.reset()
for step in range(4000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    sleep(env.envs[0].dt)

    if done[0]:
        print(f"Episode ended at step {step}. Resetting...")
        obs = env.reset()

    # Camera tracking on pelvis
    if step == 0:
        viewer = env.envs[0].unwrapped.mujoco_renderer.viewer
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.envs[0].model.body("pelvis").id

env.close()
