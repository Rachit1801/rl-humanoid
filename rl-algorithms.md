### What is MPC?

Model Predictive Control (MPC) uses a mathematical model of the robot, predicts future states and executes the best action

Example : MPC predicts where the robot’s center of mass (CoM) will move then it computes joint torques, footstep positions, body motion to maintain balance and stable walking.

The controller tries to keep the ZMP (Zero Moment Point) inside the support polygon (foot area) to avoid falling.

| MPC                       | Reinforcement Learning     |
| ------------------------- | -------------------------- |
| Model-based               | Data-driven                |
| Uses physics equations    | Learns through experience  |
| Stable and predictable    | Flexible and adaptive      |
| Easier safety constraints | Harder to guarantee safety |


