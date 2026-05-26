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


