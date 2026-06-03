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
