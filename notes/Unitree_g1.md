# Unitree G1

<img src="https://shop.deltarobots.com/cdn/shop/files/a0e8042bd2df4c47af28ff371bb6b4fa_2740x1720_cd503040-a0f7-436b-bacd-183582998e45.jpg?v=1744201010" alt="Unitree G1 – Delta Robots" style="zoom:20%;" />

To train a robot with 29 degree of freedom we can directly control torque of all 29 motors. But the problem is the robot becomes very hard to train and a random policy does things like `left_knee = +139 Nm
right_knee = -139 Nm`

### Desired joint position (PD Control)

Instead of applying 50 Nm you move knee to 0.5 radians. Then a controller computes the torque automatically.

Example:

Current knee : `q = 0.2 rad`
Desired knee: `q_target = 0.5 rad`
Error: `0.5 - 0.2 = 0.3`
`qvel = 0.1`

Simple PD controller is `torque = kp * (q_target - q) - kd * qvel` (_kp = stiffness kd = damping_)

Choose: `kp = 100 kd = 5`. 
Then `torque = 100*(0.5-0.2) - 5*(0.1) = 29.5 Nm`

Think of it like a spring. If your knee is away from the target torque increase. If your knee reaches the target  torque decrease. If your knee moves too fast damping slows it down.

For humanoids, PPO usually outputs `desired_joint_positions` not torques. Then PD control converts those positions into torques. This is why modern locomotion training is much more stable.

As the actuators of G1 uses pure torque actuators `self.data.ctrl[i]` is interpreted as apply this much torque. There is no built-in position controller.

Now to get target knee, we create reference standing position. For example:

```
target_q = np.zeros(29)

target_q[0] = -0.2    # left hip pitch
target_q[3] =  0.4    # left knee
target_q[4] = -0.2    # left ankle pitch

target_q[6] = -0.2    # right hip pitch
target_q[9] =  0.4    # right knee
target_q[10]= -0.2    # right ankle pitch
```

These numbers are just a starting guess.

In PD control, we define a target position for each joint. The controller continuously applies torque to reduce the error between the current joint position and the target joint position.

If the joint is far from the target, the controller applies a larger corrective torque. As the joint approaches the target, the corrective torque decreases. The derivative term adds damping, reducing oscillations and preventing overshoot.

This raises an important question, What happens if the chosen target joint configuration is not actually a stable pose for the robot? Can a PD controller still keep the robot balanced, or will the robot eventually fall despite perfectly tracking the target pose?

`target_q = standing_pose + 0.15 * action` The `action` comes from PPO. So PPO is learning values of action. We are not training RL to produce motor torques anymore. We are training RL to produce desired joint positions (or offsets from a standing pose), and PD handles the motor control underneath.

The PD controller handles low-level motor control, while PPO learns the higher-level balancing strategy.

I found official kp and kd values in https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/robots/g1_29dof/config/config.yaml 

---

### xml code of the g1

``` xml
<motor name="left_hip_pitch" joint="left_hip_pitch_joint" ctrlrange="-88 88" />
<motor name="left_hip_roll" joint="left_hip_roll_joint" ctrlrange="-88 88" />
...
```

cltr range is different for each joint

``` python
TORQUE_LIMITS = np.array([
    # Left leg  (6 joints)
    88, 88, 88, 139, 50, 50,      # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    # Right leg (6 joints)
    88, 88, 88, 139, 50, 50,
    # Waist     (3 joints)
    88, 50, 50,                    # yaw, roll, pitch
    # Left arm  (7 joints)
    25, 25, 25, 25, 25, 5, 5,     # shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
    # Right arm (7 joints)
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)               # shape (29,)

STANDING_HEIGHT = 0.793   # pelvis z at startup (from XML:  pos="0 0 0.793")

```

The G1 has a "free joint" at the pelvis (6 DOF, not actuated) plus 29 actuated joints.

qpos layout (36 values total):

- **[0] :  x  position of pelvis**  : we SKIP this (irrelevant for standing)
- **[1] : y  position of pelvis**  : we SKIP this
- **[2] : z  (height)**  : keep: tells us if we are standing
- **[3:7] : quaternion (w,x,y,z)** : keep: tells us our tilt
- **[7:36] : 29 joint angles**  : keep: tells us body configuration

qvel layout (35 values total):

- **[0:3] : linear velocity  (vx, vy, vz)**
- **[3:6] : angular velocity (wx, wy, wz)** : helpful to detect tipping
- **[6:35] : 29 joint velocities**

Total observation:  qpos[2:] = 34  +  qvel = 35 = 69 dimensions

#### Reward 

1. **HEIGHT REWARD**

​	How high is the pelvis compared to the target. Clipped to [0, 1] so it can't go negative if we briefly dip.

​	`height_reward = float(np.clip(height / STANDING_HEIGHT, 0.0, 1.0`

2. **UPRIGHT REWARD**

​	`data.body("pelvis").xmat` is the 3×3 rotation matrix of the pelvis. stored as 9 values, row-major).

​	`xmat[8] = R[2,2] = dot(body_z_axis, world_z_axis)`

​	1.0  when perfectly vertical
​	0.0  when tipped 90°
​	-1.0 when upside-down

​	This is the same intuition as the pole_angle term in CartPole, just expressed as "how much does the robot's up-direction agree with the world's up-direction".

​	`upright = float(self.data.body("pelvis").xmat[8])`

3. **ENERGY PENALTY**

​	Penalise large actions (normalised). Stops the robot from learning to flail wildly same `0.01 * action^2 `idea you used in CartPole.

​	`energy_penalty = 0.001 * float(np.sum(action ** 2))`

​    4. **SURVIVAL BONUS**

​	A small constant reward for every step the robot is still alive. This directly incentivises "don't fall" as the primary objective.

​	`survival_bonus = 0.5`

Hence

``` python
reward = (
              height_reward * 2.0   # stay tall            (max  2.0 per step)
            + upright       * 2.0   # stay vertical        (max  2.0 per step)
            + survival_bonus        # don't fall           (max  0.5 per step)
            - energy_penalty        # don't waste energy   (always negative)
        )
```

#### Termination

``` python
terminated = bool(
            height  < 0.35   # pelvis below 35 cm  → on the ground
            or upright < 0.5  # tilted > ~60° from vertical → unrecoverable
        )
```

---

### My Story while Training the model

A custom Gymnasium environment was developed for the Unitree G1 humanoid robot in MuJoCo. Initially, PPO directly controlled the motor torques, but this resulted in unstable simulations due to large random torques generated during early training. To improve stability, the control architecture was changed to use a PD (Proportional-Derivative) controller.

A nominal standing pose was defined using hip, knee, and ankle joint targets. Instead of generating torques directly, PPO now outputs small offsets to these target joint positions. The PD controller converts the desired joint positions into torques, allowing PPO to focus on high-level balance control while the PD controller handles low-level motor control.

The PD controller was tested independently without reinforcement learning. The robot was able to maintain its standing pose for approximately 1200 simulation steps before eventually toppling over. This confirmed that the controller was stable and no longer produced simulation explosions or NaN errors. However, the robot still lacks an active balancing strategy, as a PD controller can maintain a pose but cannot reason about balance, center of mass, or falling direction.

---

I looked at source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py and it was a pure issac labs walking code. I found a few observations

I used currently use qpos, qvel as observation but Unitree uses 
base angular velocity (3)
projected gravity (3)
velocity command (3)
joint positions (12)
joint velocities (12)
previous action (12)
sin phase (1)
cos phase (1)

They explicitly penalize vibration with joint_vel, action_rate, joint_acc, energy. These are all vibration killers.

They train with pushes every 5 seconds. So the robot learns recovery from disturbances.

They terminate much earlier `bad_orientation(limit_angle=0.8)` 0.8 rad ≈ 46°

Rewards

```
velocity tracking
upright reward
energy penalty
action rate penalty
joint limit penalty
feet clearance
feet contact
```

Also I tried using action smoother but unitree dosnet use it

They use recurrent PPO `rnn_type = "lstm"`. It means the policy remembers recent history.

---

After training and working graphs

#### train/approx_kl

Shows PPO stability.

![image-20260614113657060](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20260614113657060.png)

After approximately 3 million training timesteps, the learned policy achieved an average episode length of ~1652 simulation steps and an average reward of ~6309. The value function converged successfully with an explained variance of 0.988 and a value loss close to zero, indicating stable PPO optimization and successful acquisition of the standing behavior.

#### rollout/ep_len_mean

Average episode length during the rollout.

![image-20260614114657273](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20260614114657273.png)

#### rollout/ep_rew_mean

Average total reward obtained per episode. PPO tries to maximize this value.

![image-20260614114808237](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20260614114808237.png)

# Balance on a Moving Platform

Plan 

1) Stand on ground
2) Stand while being pushed
3) Stand on a moving platform
4) Stand on a moving platform with unpredictable disturbances

Every few seconds Apply random pushes to pelvis. `force_x = np.random.uniform(-20, 20)`

Slowly increase the random push force

