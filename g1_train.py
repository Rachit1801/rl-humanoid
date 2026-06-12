"""
G1 Humanoid Standing — Parallel PPO Training Script
====================================================

Trains the Unitree G1 29-DOF humanoid to stand using PPO with:
  - 8 parallel environments (SubprocVecEnv)
  - Observation & reward normalization (VecNormalize)
  - TensorBoard logging with episode rewards
  - Periodic model checkpointing

Usage:
    1. Activate virtual env:  rl\\Scripts\\activate
    2. Run:                   python g1_train.py
    3. Monitor TensorBoard:   tensorboard --logdir=./tb_logs/

    To resume training from a checkpoint:
        Uncomment the "Resume training" section and update the path.

Output:
    models/g1_stand_v2.zip            — Trained PPO model
    models/g1_stand_v2_vecnorm.pkl    — VecNormalize statistics (needed for inference)
    models/checkpoints/               — Periodic checkpoints
    tb_logs/g1_stand/                 — TensorBoard logs
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)

from envs.g1_stand_env import G1StandEnv, make_env


# ─── Configuration ────────────────────────────────────────────────────────────
NUM_ENVS         = 8              # Number of parallel environments
TOTAL_TIMESTEPS  = 3_000_000      # Total training timesteps (increased for 2000-step episodes)
SAVE_DIR         = "models"       # Directory for saved models
TB_LOG_DIR       = "./tb_logs/"   # TensorBoard log directory
CHECKPOINT_FREQ  = 50_000         # Save checkpoint every N timesteps


# ─── Custom Callback for TensorBoard Logging ─────────────────────────────────

class EpisodeRewardCallback(BaseCallback):
    """
    Logs episode-level metrics to TensorBoard.

    Tracks raw (un-normalized) episode rewards and lengths from VecMonitor
    and logs rolling averages for easy monitoring.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        # VecMonitor stores completed episode info in 'infos'
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self._episode_rewards.append(ep_reward)
                self._episode_lengths.append(ep_length)

        # Log rolling averages every 2048 steps (once per rollout roughly)
        if self.n_calls % 2048 == 0 and len(self._episode_rewards) > 0:
            recent_rewards = self._episode_rewards[-100:]
            recent_lengths = self._episode_lengths[-100:]

            self.logger.record(
                "episode/mean_reward", np.mean(recent_rewards)
            )
            self.logger.record(
                "episode/mean_length", np.mean(recent_lengths)
            )
            self.logger.record(
                "episode/max_reward", np.max(recent_rewards)
            )
            self.logger.record(
                "episode/min_reward", np.min(recent_rewards)
            )
            self.logger.record(
                "episode/total_episodes", len(self._episode_rewards)
            )

        return True


# ─── Main Training Loop ──────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Environment Sanity Check ──────────────────────────────────────
    print("=" * 60)
    print("  G1 Humanoid Standing — PPO Training")
    print("=" * 60)
    print("\n[1/4] Checking environment validity...")

    test_env = G1StandEnv()
    check_env(test_env, warn=True)
    del test_env
    print("  ✓ Environment check passed!\n")

    # ── Step 2: Create Parallel Training Environments ─────────────────────────
    print(f"[2/4] Creating {NUM_ENVS} parallel environments...")

    # SubprocVecEnv: each env runs in its own process
    train_env = SubprocVecEnv(
        [make_env(rank=i, seed=42) for i in range(NUM_ENVS)]
    )

    # VecMonitor: tracks episode rewards & lengths (before normalization)
    train_env = VecMonitor(train_env)

    # VecNormalize: normalizes observations and rewards for stable training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    print(" Environments ready!\n")

    # ── Step 3: Create PPO Model ──────────────────────────────────────────────
    print("[3/4] Initializing PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        # ── Core PPO hyperparameters ──
        learning_rate=3e-4,
        n_steps=2048,               # Steps per env before update
        batch_size=512,             # Minibatch size for SGD
        n_epochs=10,                # SGD epochs per update
        gamma=0.99,                 # Discount factor
        gae_lambda=0.95,            # GAE lambda
        clip_range=0.2,             # PPO clip range
        ent_coef=0.005,             # Entropy bonus (exploration)
        vf_coef=0.5,               # Value function loss coefficient
        max_grad_norm=0.5,         # Gradient clipping
        # ── Network architecture ──
        policy_kwargs=dict(
            net_arch=[256, 256],    # Two hidden layers, 256 units each
        ),
        # ── Logging ──
        verbose=1,
        tensorboard_log=TB_LOG_DIR,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  ✓ Model created ({total_params:,} parameters)")
    print(f"    Network: [67] → [256] → [256] → [29]")
    print(f"    Steps per update: {2048 * NUM_ENVS:,} timesteps\n")

    # ── Step 4: Callbacks ─────────────────────────────────────────────────────
    # Checkpoint: save model + VecNormalize stats periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVS, 1),
        save_path=os.path.join(SAVE_DIR, "checkpoints"),
        name_prefix="g1_stand",
        save_vecnormalize=True,
        verbose=1,
    )

    # Episode reward logging
    episode_callback = EpisodeRewardCallback()

    callbacks = CallbackList([checkpoint_callback, episode_callback])

    # ── Train! ────────────────────────────────────────────────────────────────
    print("[4/4] Starting training...")
    print(f"  Parallel envs:    {NUM_ENVS}")
    print(f"  Total timesteps:  {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoint every: {CHECKPOINT_FREQ:,} steps")
    print(f"  TensorBoard:      {TB_LOG_DIR}")
    print("=" * 60 + "\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="g1_stand",
    )

    # ── Save Final Model ──────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path = os.path.join(SAVE_DIR, "g1_stand_v2")
    vecnorm_path = os.path.join(SAVE_DIR, "g1_stand_v2_vecnorm.pkl")

    model.save(model_path)
    train_env.save(vecnorm_path)

    print("\n" + "=" * 60)
    print("  ✓ Training complete!")
    print(f"  Model saved:        {model_path}.zip")
    print(f"  VecNormalize saved: {vecnorm_path}")
    print("=" * 60)

    train_env.close()


    # ── Resume Training (uncomment to continue from checkpoint) ───────────────
    # """
    # checkpoint = "models/checkpoints/g1_stand_XXXXX_steps"
    # vecnorm_checkpoint = "models/checkpoints/g1_stand_XXXXX_steps_vecnormalize.pkl"
    #
    # train_env = SubprocVecEnv([make_env(i, seed=42) for i in range(NUM_ENVS)])
    # train_env = VecMonitor(train_env)
    # train_env = VecNormalize.load(vecnorm_checkpoint, train_env)
    #
    # model = PPO.load(checkpoint, env=train_env)
    # model.learn(total_timesteps=1_000_000, reset_num_timesteps=False,
    #             callback=callbacks, progress_bar=True, tb_log_name="g1_stand")
    # model.save("models/g1_stand_v2")
    # train_env.save("models/g1_stand_v2_vecnorm.pkl")
    # train_env.close()
    # """
