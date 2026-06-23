"""
train_walk.py  — Train the G1 walking policy using PPO.

Uses the G1WalkEnv (MuJoCo velocity-tracking env translated from Isaac Lab).
Starts from scratch with a [512, 256, 128] MLP, 8 parallel envs.

Usage:
    python train_walk.py
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)

from envs.g1_walk_env import G1WalkEnv, make_walk_env

# ──────────────────────── hyper-parameters ────────────────────────
NUM_ENVS         = 8
TOTAL_TIMESTEPS  = 30_000_000     # adjust as needed (walking takes a lot of steps)
LEARNING_RATE    = 3e-4
N_STEPS          = 4096           # rollout length per env  (large for diverse experience)
BATCH_SIZE       = 512
N_EPOCHS         = 10
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP_RANGE       = 0.2
ENT_COEF         = 0.01          # mild entropy bonus for exploration
VF_COEF          = 0.5
MAX_GRAD_NORM    = 1.0

NET_ARCH         = [512, 256, 128]

SAVE_DIR         = "models"
TB_LOG_DIR       = "./tb_logs/"
MODEL_NAME       = "g1_walk_v1"
NORM_NAME        = "g1_walk_norm_v1.pkl"
CHECKPOINT_FREQ  = max(100_000 // NUM_ENVS, 1)


# ──────────────────────── curriculum logger callback ────────────────────────
class CurriculumLoggerCallback(BaseCallback):
    """Log curriculum level to TensorBoard every N steps."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Read curriculum_level from the first env's info
        infos = self.locals.get("infos", [])
        if infos:
            level = infos[0].get("curriculum_level", 0.0)
            self.logger.record("curriculum/level", level)
        return True


# ──────────────────────── main ────────────────────────
if __name__ == "__main__":

    # ── sanity check (single env) ──
    print("Running env check …")
    check_env(G1WalkEnv(), warn=True)
    print("Env check passed\n")

    # ── vectorised envs ──
    train_env = SubprocVecEnv([make_walk_env(i) for i in range(NUM_ENVS)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    # ── PPO ──
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        policy_kwargs=dict(net_arch=NET_ARCH),
        tensorboard_log=TB_LOG_DIR,
    )

    # ── callbacks ──
    os.makedirs(os.path.join(SAVE_DIR, "walk_checkpoints"), exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(SAVE_DIR, "walk_checkpoints"),
        name_prefix="walk_ckpt",
        save_vecnormalize=True,
        verbose=1,
    )
    curriculum_cb = CurriculumLoggerCallback()
    callbacks = CallbackList([checkpoint_cb, curriculum_cb])

    # ── train ──
    print(f"\nStarting PPO training  ({TOTAL_TIMESTEPS:,} timesteps, {NUM_ENVS} envs)")
    print(f"Net arch: {NET_ARCH},  obs dim: 480,  act dim: 29\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="g1_walk",
    )

    # ── save ──
    model.save(os.path.join(SAVE_DIR, MODEL_NAME))
    train_env.save(os.path.join(SAVE_DIR, NORM_NAME))
    print(f"\nModel saved to  {SAVE_DIR}/{MODEL_NAME}")
    print(f"VecNormalize saved to  {SAVE_DIR}/{NORM_NAME}")

    train_env.close()
