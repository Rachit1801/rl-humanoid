"""
G1 Humanoid Standing — Inference / Visualization Script
========================================================

Loads a trained PPO model and renders the G1 humanoid standing
in the MuJoCo viewer.

Usage:
    1. Activate virtual env:  rl\\Scripts\\activate
    2. Run:                   python g1_run.py

Prerequisites:
    - Trained model at models/g1_stand_v2.zip
    - VecNormalize stats at models/g1_stand_v2_vecnorm.pkl
    (both are created by g1_train.py)
"""

import os
from time import sleep
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.g1_stand_env import G1StandEnv


# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH    = "models/g1_stand_v2"
VECNORM_PATH  = "models/g1_stand_v2_vecnorm.pkl"
NUM_EPISODES  = 5
MAX_STEPS     = 2000
RENDER_DELAY  = 0.01   # Seconds between frames (for smooth playback)


if __name__ == "__main__":
    print("=" * 60)
    print("  G1 Humanoid Standing — Inference")
    print("=" * 60)

    # ── Load Environment ──────────────────────────────────────────────────────
    print("\nLoading environment...")

    # DummyVecEnv wraps the single env for VecNormalize compatibility
    env = DummyVecEnv([lambda: G1StandEnv(render_mode="human")])

    # Load observation normalization statistics from training
    if os.path.exists(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False       # Don't update running stats during inference
        env.norm_reward = False    # Don't normalize rewards during inference
        print("  ✓ VecNormalize stats loaded")
    else:
        print(f"  ⚠ No VecNormalize stats found at {VECNORM_PATH}")
        print("    Running without observation normalization.")

    # ── Load Model ────────────────────────────────────────────────────────────
    print("Loading trained model...")
    model = PPO.load(MODEL_PATH)
    print(f"  ✓ Model loaded from {MODEL_PATH}.zip\n")

    # ── Run Episodes ──────────────────────────────────────────────────────────
    print(f"Running {NUM_EPISODES} episodes (max {MAX_STEPS} steps each)...\n")

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        for step in range(MAX_STEPS):
            # Get action from trained policy (deterministic = no exploration noise)
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_steps += 1

            # Render
            env.render()
            sleep(RENDER_DELAY)

            if done[0]:
                height = info[0].get("height", "N/A")
                print(
                    f"  Episode {ep + 1}/{NUM_EPISODES}  |  "
                    f"Steps: {episode_steps:4d}  |  "
                    f"Reward: {episode_reward:8.1f}  |  "
                    f"Height: {height:.3f}m  |  "
                    f"{'FELL' if episode_steps < 1000 else 'SURVIVED'}"
                )
                break
        else:
            print(
                f"  Episode {ep + 1}/{NUM_EPISODES}  |  "
                f"Steps: {episode_steps:4d}  |  "
                f"Reward: {episode_reward:8.1f}  |  "
                f"COMPLETED"
            )

    env.close()
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)
