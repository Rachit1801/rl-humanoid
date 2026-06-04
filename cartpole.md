### Gymnasium

Gymnasium is a python framework/API for creating and interacting with reinforcement learning environments.

| Term      | Analogy                          |
| --------- | -------------------------------- |
| Simulator | Engine                           |
| PPO       | RL Algorithm                     |
| Gymnasium | Interface between RL and Physics |
| MuJoCo    | Simulates Physics                |

MuJoCo computes how the body physically moves using physics and gymnasium wraps this into obs, reward, terminated, truncated, info for RL training.

---

### Stable-Baselines3

Stable-Baselines3 provides ready-made RL algorithms. Like PPO, SAC, TD3

---

### Isaac Lab

Isaac Lab is a robotics RL framework from NVIDIA. It combines robotics, simulation, RL training, large-scale parallel environments. Used heavily for humanoids, quadrupeds, manipulation.

---

### PPO

Proximal Policy Optimization Developed by OpenAI

PPO is a reinforcement learning algorithm that learns a policy (behavior) while preventing updates from changing too drastically at once.

PPO trains a neural network so actions that give higher rewards become more likely.

Example : Reward +1 for staying upright, -10 for falling

**Major PPO Libraries**

| Library               | Primary Framework    | Target Use-Case                   |
| --------------------- | -------------------- | --------------------------------- |
| **Stable-Baselines3** | PyTorch              | Reinforcement Learning & Robotics |
| **Hugging Face TRL**  | PyTorch              | Generative AI, LLMs & RLHF        |
| **Ray RLlib**         | TensorFlow / PyTorch | Distributed Systems & Production  |
| **TorchRL**           | PyTorch              | Advanced Research / Custom Loops  |

| Parameter     | Meaning                  |
| ------------- | ------------------------ |
| learning_rate | update speed             |
| gamma         | future reward importance |
| batch_size    | training chunk size      |
| n_steps       | rollout length           |

---

```xml
<mujoco>
    <worldbody>
        <!-- Ground -->
        <geom type="plane" size="5 5 0.1"/>
        <!-- Rail (For Refrence)-->
        <geom type="box" pos="0 0 0" size="2 0.05 0.05" rgba="0.3 0.3 0.3 1"/>
        <!-- Cart -->
        <body pos="0 0 0.1">
            <!-- Cart moves left/right -->
            <joint type="slide" axis="1 0 0"/>
            <!-- Cart body -->
            <geom type="box" size="0.2 0.15 0.1" rgba="1 0 0 1"/>
            <!-- Pole -->
            <body pos="0 0 0.1">
                <!-- Pole rotates -->
                <joint type="hinge" axis="0 1 0"/>
                <!-- Pole geom -->
                <geom type="capsule" fromto="0 0 0 0 0 1" size="0.05" rgba="0 1 0 1"/>
            </body>
        </body>
    </worldbody>
</mujoco>
```

```python
import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class CartPoleEnv(gym.Env):

    def __init__(self):

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("cartpole.xml")
        self.data = mujoco.MjData(self.model)

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        mujoco.mj_resetData(self.model, self.data)

        obs = self._get_obs()

        return obs, {}

    def step(self, action):

        # Apply action
        self.data.ctrl[0] = action[0]

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Observation
        obs = self._get_obs()

        # Reward
        reward = 1.0

        # Termination condition
        pole_angle = self.data.qpos[1]

        terminated = abs(pole_angle) > 0.5

        truncated = False

        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):

        return np.array([
            self.data.qpos[0],   # cart position
            self.data.qvel[0],   # cart velocity
            self.data.qpos[1],   # pole angle
            self.data.qvel[1]    # pole angular velocity
        ], dtype=np.float32)
```

---

# Gymnasium

```python
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

# Start a fresh episode
observation, info = env.reset(seed=42)
print(f"Initial obs: {observation}")
# [0.0, 0.0, 0.03, -0.02] — nearly upright

total_reward = 0.0
step_count = 0

# Keep stepping until episode ends
while True:
    # Step 1: Choose an action
    action = env.action_space.sample()   # random action for now

    # Step 2: Apply action, get results
    observation, reward, terminated, truncated, info = env.step(action)

    # Step 3: Accumulate reward
    total_reward += reward
    step_count += 1

    # Step 4: Print what happened
    print(f"Step {step_count}: action={action}, reward={reward:.2f}, "
          f"obs={observation[:2]}, terminated={terminated}, truncated={truncated}")

    # Step 5: Check if episode is over
    if terminated or truncated:
        print(f"\nEpisode ended after {step_count} steps. Total reward: {total_reward}")
        break

env.close()
```

---

```python
from time import sleep
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv #MujocoEnv(wrapper) internally handles the MuJoCo import
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class MyCartPoleEnv(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        super().__init__(model_path="C:/Users/admin/Desktop/rl-humanoid/cartpole.xml", frame_skip=1, observation_space=observation_space, render_mode=render_mode)

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):

        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset_model(self):

        self.set_state(qpos=np.array([0.0, 0.05]), qvel=np.array([0.0, 0.0]))
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()
        cart_pos = obs[0]
        pole_angle = obs[1]
        reward = 1.0
        terminated = bool(abs(cart_pos) > 2.0 or abs(pole_angle) > 0.5)
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)

train_env = MyCartPoleEnv(render_mode=None)
check_env(train_env)                          # Check Env (one time only)

model = PPO(policy="MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, verbose=1)

# Load and continue training
# model = PPO.load("ppo_cartpole", env=train_env)
# model.learn(total_timesteps=50_000, reset_num_timesteps=False)
# model.save("ppo_cartpole_v2")

print("\nStarting PPO training...")
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")
train_env.close()

env = MyCartPoleEnv(render_mode="human")
obs, info = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Step: {step} Observation: {obs} Reward: {reward}")
    sleep(0.02)
    if terminated or truncated:
        print("Episode ended at step {step}. Resetting...\n")
        obs, info = env.reset()
env.close()
```

| Parameter       | What it controls                 | Try if struggling        |
| --------------- | -------------------------------- | ------------------------ |
| `learning_rate` | Step size for gradient updates   | Lower to `1e-4`          |
| `n_steps`       | How much data before each update | Increase to `4096`       |
| `gamma`         | How much to value future rewards | Keep at `0.99`           |
| `n_epochs`      | How many passes over each batch  | Lower to `5` if unstable |

---

### To resume training later

```python
# Load and continue training
model = PPO.load("ppo_cartpole", env=train_env)
model.learn(total_timesteps=50_000, reset_num_timesteps=False)
model.save("ppo_cartpole_v2")
```

![](C:\Users\admin\AppData\Roaming\marktext\images\2026-06-04-23-25-52-image.png)
