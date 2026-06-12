import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from envs.g1_env import G1Env
from envs.g1_env import make_env

num_env = 8

if __name__ == "__main__":      #Windows Guard(only needed in Windows)
    check_env(G1Env(), warn=True)                          # Check Env (one time only)
    print("Env Check SuccessFul")
    train_env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    # train_env = MyCartPoleEnv(render_mode=None)       # For Single Training
    train_env = VecMonitor(train_env)   # tracks episode rewards & lengths
    train_env = VecNormalize(train_env,norm_obs=True,norm_reward=True,clip_obs=10.0,clip_reward=10.0,gamma=0.99,)

    model = PPO(policy="MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10, gamma=0.99, verbose=0, tensorboard_log = "./tb_logs/")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // 8, 1),
        save_path=os.path.join("models", "checkpoints"),
        name_prefix="g1_stand",
        save_vecnormalize=True,
        verbose=1,
    )

    callbacks = CallbackList(checkpoint_callback)

    print("\nStarting PPO training...")
    model.learn(total_timesteps=2_000_000,callback=callbacks, progress_bar=True)
    model.save("models/g1_stand")
    print("\nTraining Complete")
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