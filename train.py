import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.g1_env import G1Env
from envs.g1_env import make_env

num_env = 8

if __name__ == "__main__":      #Windows Guard(only needed in Windows)
    train_env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    # train_env = MyCartPoleEnv(render_mode=None)       # For Single Training
    check_env(G1Env())                          # Check Env (one time only)
    print("Env Check SuccessFul")

    model = PPO(policy="MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, verbose=0) # tensorboard_log = "./tb_logs/"

    # Load and continue training
    # model = PPO.load("models/training_data", env=train_env)
    # model.learn(total_timesteps=50_000, reset_num_timesteps=False)
    # model.save("ppo_cartpole_v2")

    print("\nStarting PPO training...")
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save("models/g1_stand")
    print("\nTraining Complete")
    train_env.close()
