### Gymnasium

Gymnasium is a python framework/API for creating and interacting with reinforcement learning environments.

| Term      | Analogy                                |
| --------- | -------------------------------------- |
| Simulator | Game                                   |
| PPO       | player/AI                              |
| Gymnasium | how the player interacts with the game |
| MuJoCo    | Simulates Physics                      |

MuJoCo computes how the body physically moves using physics and gymnasium wraps this into obs, reward, terminated, truncated, info for RL training.

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
