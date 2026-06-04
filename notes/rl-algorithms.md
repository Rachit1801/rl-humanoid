### What is MPC?

Model Predictive Control (MPC) uses a mathematical model of the robot, predicts future states and executes the best action

Example : MPC predicts where the robot’s center of mass (CoM) will move then it computes joint torques, footstep positions, body motion to maintain balance and stable walking.

The controller tries to keep the ZMP (Zero Moment Point) inside the support polygon (foot area) to avoid falling.

| MPC | Reinforcement Learning |
| --- | --- |
| Model-based | Data-driven |
| Uses physics equations | Learns through experience |
| Stable and predictable | Flexible and adaptive |
| Easier safety constraints | Harder to guarantee safety |

### What is WBC?

Whole-Body Control (WBC) controls the entire humanoid robot body simultaneously instead of controlling each joint independently.

WBC coordinates ams, legs, torso, head, balance all together in one unified framework.

### What is Imitation Learning?

Imitation Learning is a machine learning method where a robot learns by copying expert behavior instead of learning completely through trial and error.

Example : A humanoid robot observes human motion capture data, expert walking controllers or recorded trajectories and learns walking, balancing, running or other movements by imitating them.

Imitation learning helps humanoid robots learn faster, produce more natural motion and reduce unstable exploration during training.

---

# Open Source Humanoids

Note : I have only included projects which used Reinforcement learning as it is part of my internship

### Berkeley Humanoid Lite

- Organization : UC Berkeley Hybrid Robotics Lab
- Description : Low-cost open-source 3D-printed humanoid robot designed for reinforcement learning and sim-to-real research.
- Main Research Focus : Reinforcement learning locomotion and zero-shot sim-to-real transfer
- RL Algorithm: Proximal Policy Optimization (PPO) (utilizing LeggedGym environments for parallelized simulation).
- Paper :
  - https://arxiv.org/abs/2504.17249
  - https://www.roboticsproceedings.org/rss21/p062.pdf
- Code : https://github.com/hybridrobotics/berkeley-humanoid-lite
- Simulation : IsaacLab, Isaac Gym
- Programming Stack : Python, C++

### Cassie

- Organization : Agility Robotics
- Description : Highly dynamic biped robot with bio-inspired legs widely used in academic RL locomotion research.
- Main Research Focus : Reinforcement learning for robust locomotion
- RL Algorithm: Proximal Policy Optimization (PPO) (some early work also explored Soft Actor-Critic (SAC) and Deep Deterministic Policy Gradient (DDPG)).
- Paper :
  - https://arxiv.org/abs/2103.14295
  - https://proceedings.mlr.press/v100/xie20a.html
  - https://arxiv.org/abs/1803.05580
- Code : https://github.com/HybridRobotics/cassie_rl_walking
- Simulation : MuJoCo, Gazebo, Simulink, PyBullet
- Programming Stack : Python

### MEVITA

- Organization : University of Tokyo / Kento Kawaharazuka Research Group
- Description : Open-source bipedal humanoid robot built using e-commerce components and sheet metal welding for low-cost robust locomotion research.
- Main Research Focus : Reinforcement learning locomotion and sim-to-real transfer
- RL Algorithm: Proximal Policy Optimization (PPO) (specifically utilizing the rsl_rl library and LeggedGym).
- Paper :
  - https://arxiv.org/html/2508.17684v1
  - https://www.semanticscholar.org/paper/MEVITA%3A-Open-Source-Bipedal-Robot-Assembled-From-Kawaharazuka-Sawaguchi/6dfdb11eaca5e0cd5d94a12475a7b3aa55b8e76b
- Code : https://github.com/haraduka/mevita
- Simulation : MuJoCo
- Programming Stack : ROS, Python, C++

---

# Open Source Physics Simulator

### MuJoCo by Google DeepMind

High-fidelity physics simulator widely used for humanoid robotics and reinforcement learning research.

Code : [GitHub - google-deepmind/mujoco: Multi-Joint dynamics with Contact. A general purpose physics simulator. · GitHub](https://github.com/google-deepmind/mujoco)

Programming Stack : C++, Python

---

## Reinforcement Learning Workflow for Humanoid Robots

1. **Policy Training** : Neural network learns behaviors from reward signals.
2. **Simulation (MuJoCo)** : Physics-based testing of locomotion policies.
3. **Transfer to Hardware** : Zero-shot sim-to-real deployment.
4. **Hardware Testing** : Validating real-world performance on the robot.

![RL workflow for humanoid robots](https://copilot.microsoft.com/th/id/BCO.5b70b472-d68f-4e6c-8e5c-f371e00cf63d.png)
