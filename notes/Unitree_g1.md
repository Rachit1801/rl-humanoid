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

---

Now in the xml code of the g1

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

