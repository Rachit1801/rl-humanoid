"""
Train Unitree G1 to walk on a moving platform with curriculum learning.

Uses SB3 PPO with an Asymmetric Actor-Critic policy.
Each parallel environment independently promotes/demotes its own curriculum
level based on episode survival, similar to the official terrain_levels_vel().

Network architecture follows the official rl_cfg.py: (512, 256, 128) with ELU.
"""

import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.policies import ActorCriticPolicy

from envs.g1_walk_env import G1WalkEnv, make_env
from envs.g1_walk_config import TOTAL_TIMESTEPS, NUM_CURRICULUM_STAGES, ACTOR_OBS_DIM


# ═══════════════════════════════════════════════════════════════════════════════
# Asymmetric Actor-Critic Policy
#
# The observation is a flat 115D array:
#   indices 0:98   = actor features (ang_vel, gravity, cmd, phase, joint_pos/vel, last_action)
#   indices 98:115 = critic-only features (lin_vel, platform_vel, foot_height/air/contact/forces)
#
# The actor sees zeros for indices 98:, the critic sees everything.
# ═══════════════════════════════════════════════════════════════════════════════

_CRITIC_START = ACTOR_OBS_DIM  # = 98

class AsymmetricPolicy(ActorCriticPolicy):
    def extract_features(self, obs, features_extractor=None):
        features = super().extract_features(obs, features_extractor)
        
        # When share_features_extractor=False, SB3 returns a tuple (pi_features, vf_features)
        if isinstance(features, tuple):
            pi_features, vf_features = features
            # Hide critic-only features (last 17 dims) from the Actor
            masked_pi = pi_features.clone()
            masked_pi[:, _CRITIC_START:] = 0.0
            return masked_pi, vf_features
        
        # Fallback for share_features_extractor=True
        if features_extractor is self.pi_features_extractor or features_extractor is None:
            masked = features.clone()
            masked[:, _CRITIC_START:] = 0.0
            return masked
            
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Logger callback
#
# The curriculum is step-based: velocity ranges expand after
# CURRICULUM_VEL_EXPAND_STEP total steps. This callback logs progress.
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumLoggerCallback(BaseCallback):
    """Periodically log the curriculum level of each parallel env."""

    def __init__(self, log_interval: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            levels = self.training_env.env_method("get_curriculum_level")
            if self.verbose:
                level_str = "  ".join(
                    f"env{i}:L{lv}" for i, lv in enumerate(levels)
                )
                avg = sum(levels) / len(levels)
                print(
                    f"  [{self.num_timesteps:>10,} steps]  "
                    f"Curriculum levels: {level_str}  "
                    f"(avg {avg:.1f}/{NUM_CURRICULUM_STAGES - 1})"
                )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

NUM_ENVS = 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--resume_version", type=str, default="4", help="Version to resume from")
    parser.add_argument("--version", type=str, default="5", help="Version to save as")
    args = parser.parse_args()

    # ── Sanity check ────────────────────────────────────────────────────
    print("Running env check ...")
    check_env(G1WalkEnv(), warn=True)
    print("Env check passed\n")

    # ── Vectorised environment ──────────────────────────────────────────
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    train_env = VecMonitor(train_env)
    
    if args.resume:
        print(f"Resuming from models/g1_walk_norm_v{args.resume_version}.pkl ...")
        train_env = VecNormalize.load(f"models/g1_walk_norm_v{args.resume_version}.pkl", train_env)
        train_env.training = True
        train_env.norm_reward = False
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=100.0,
            gamma=0.99,
        )

    # ── PPO model ───────────────────────────────────────────────────────
    # Architecture: (512, 256, 128) ELU  — from official rl_cfg.py
    if args.resume:
        print(f"Resuming from models/g1_walk_v{args.resume_version}.zip ...")
        model = PPO.load(
            f"models/g1_walk_v{args.resume_version}.zip",
            env=train_env,
            tensorboard_log="./tb_logs/",
            # Reset learning rate to give it a fresh start if needed, but usually we just keep it
        )
    else:
        model = PPO(
            policy=AsymmetricPolicy,
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=5,                          # official: 5
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,                        # official: 0.01
            max_grad_norm=1.0,                   # official: 1.0
            verbose=1,
            policy_kwargs=dict(
                net_arch=[512, 256, 128],
                activation_fn=torch.nn.ELU,
                share_features_extractor=False,
            ),
            tensorboard_log="./tb_logs/",
        )

    # ── Callbacks ───────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // NUM_ENVS, 1),
        save_path=f"models/walk_checkpoints_v{args.version}",
        name_prefix=f"walk_ckpt_v{args.version}",
        save_vecnormalize=True,
        verbose=1,
    )

    curriculum_logger_cb = CurriculumLoggerCallback(
        log_interval=10_000,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, curriculum_logger_cb])

    # ── Train ───────────────────────────────────────────────────────────
    print(f"Starting walking training  ({TOTAL_TIMESTEPS:,} timesteps)")
    print(f"  Curriculum      : Step-based velocity expansion")
    print(f"  Stages          : {NUM_CURRICULUM_STAGES}")
    print(f"  Parallel envs   : {NUM_ENVS}")
    print(f"  Obs dims        : 115 (actor=98, critic-only=17)")
    print(f"  Network         : [512, 256, 128] ELU")
    print(f"  Actor-Critic    : Asymmetric (critic-only: lin_vel, platform, foot info)\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="g1_walk",
    )

    # ── Save ────────────────────────────────────────────────────────────
    model.save(f"models/g1_walk_v{args.version}")
    train_env.save(f"models/g1_walk_norm_v{args.version}.pkl")
    print("\nTraining complete")
    print(f"  Model        -> models/g1_walk_v{args.version}")
    print(f"  VecNormalize -> models/g1_walk_norm_v{args.version}.pkl")

    train_env.close()

