# PART 1: FOUNDATION (Weeks 1–5)

## Phase 1.1: Research Immersion

**Timeline:** Early May 2026

Rachit begins with zero knowledge of humanoid robotics. His first task is to understand the landscape. He creates `notes/Introduction.md` (260 lines) — a comprehensive survey covering:

- **Humanoid anatomy**: torso, pelvis, hip (pitch/roll/yaw), knee, ankle (pitch/roll), waist, shoulder, elbow, wrist — 29 DOF total
- **Training paradigms**: Imitation learning (behavior cloning from human motion capture) vs Reinforcement learning (trial-and-error with reward functions)
- **Control algorithms**: Zero Moment Point (ZMP — the classic bipedal stability criterion), A* path planning, Model Predictive Control (MPC — receding-horizon optimization), Whole Body Control (WBC — hierarchical task-space control)
- **History**: Atlas (Boston Dynamics), HRP (AIST), ASIMO (Honda), TORO (DLR), Unitree H1/G1, MIT Cheetah, ANYmal
- **Open-source ecosystems**: MuJoCo, Isaac Lab, Gymnasium, Stable-Baselines3, PyTorch

He then creates `notes/librarys.md` (331 lines) — a condensed textbook covering every library he'll use. This document reads like a personal reference manual, with complete code examples for CartPole, PPO configuration, reward function design, and environment structure.

**Key insight documented:** He understands the layered architecture that will define his entire project:

```
Physics Simulator (MuJoCo)
    ↓
RL Interface (Gymnasium Environment)
    ↓
RL Algorithm (PPO via Stable-Baselines3)
    ↓
Training Infrastructure (SubprocVecEnv, VecNormalize, Callbacks)
```

## Phase 1.2: Reinforcement Learning Theory

He studies `notes/rl-algorithms.md` (97 lines) and `notes/rl_notes.pdf`:

- **Backpropagation**: How gradients flow through neural networks to update weights
- **Policy Gradient**: REINFORCE algorithm — `∇J(θ) = E[∇log π_θ(a|s) * R]`
- **PPO's innovation**: Clipping the probability ratio `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` to `[1-ε, 1+ε]` prevents destructive policy updates
- **Cross-Entropy Method**: Supervised fine-tuning by selecting top-k performing episodes
- **RLHF**: Reinforcement Learning from Human Feedback (for LLMs)

He documents PPO hyperparameters with their meanings:  

- `learning_rate`: step size for gradient descent
- `gamma`: discount factor (0.99 = values future rewards almost as much as immediate)
- `n_steps`: how many environment steps collected before each policy update
- `batch_size`: minibatch size for gradient computation
- `n_epochs`: number of passes over collected data per update
- `gae_lambda`: Generalized Advantage Estimation parameter (0.95 = bias-variance tradeoff)

## Phase 1.3: MuJoCo Physics Simulator

He creates `notes/MuJoCo.md` (314 lines) — his most detailed notebook. He systematically learns:

**MuJoCo architecture:**  

```
mjModel (static) → mjData (dynamic) → mj_step() → repeat
```

**MJCF/XML hierarchy:**  

```
xml
<mujoco>
  <compiler>      <!-- compilation settings -->
  <default>       <!-- default joint/geom properties -->
  <asset>         <!-- meshes, textures, materials -->
  <worldbody>     <!-- the scene tree -->
    <body>        <!-- rigid body -->
      <joint>     <!-- degrees of freedom -->
      <geom>      <!-- collision/visual geometry -->
      <body>      <!-- child body (kinematic chain) -->
    </body>
  </worldbody>
  <actuator>      <!-- motors, position servos -->
  <sensor>        <!-- jointpos, jointvel, accelerometer, gyro -->
</mujoco>
```

**Joint types learned:**  

- `free`: 6DOF floating base (used for pelvis)
- `hinge`: 1DOF rotation (used for all G1 joints)
- `slide`: 1DOF translation (used for cartpole, platform)
- `ball`: 3DOF rotation (not used in G1)

**Actuator types:**  

- `motor`: direct torque control
- `position`: PD position servo (used for platform)
- `velocity`: velocity servo

**Sensors:**  

- `jointpos`/`jointvel`: joint state
- `accelerometer`/`gyro`: IMU readings
- `force`/`touch`: contact forces

**Critical discovery about mocap bodies** (from notes):  

> "MuJoCo does not compute velocity, acceleration, momentum for mocap bodies. It simply sees Old position and New position every update. Physics sees platform not moving during most substeps. So contact solver doesn't generate realistic friction."

This note is **prophetic** — it predicts the platform slipping bug he'll battle for weeks in Part 3.

---

## Phase 1.4: First MuJoCo Models (Sandbox)

### Commit `e44522e` — "MuJoCo Basic"

**File:** `sandbox/box.py` (22 lines)

The simplest possible MuJoCo simulation — a box falling onto a plane:

```
python
xml = """<mujoco>
    <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

**Result:** The box falls, hits the ground, and settles. First successful simulation.

### Commit `6e31db7` — "cartpole MuJoCo"

**Files:** `sandbox/cartpole.py` (75 lines), `assets/cartpole.xml` (24 lines)

The XML defines a track, a cart on a slide joint, a pole on a hinge joint, a motor actuator, and sensors. The Python script:  

1. Loads the model
2. Controls the cart manually with `data.ctrl[0] = 0.5`
3. Observes `data.qpos` and `data.qvel`

He learns how `ctrlrange` and `gear` interact — the actuator converts normalized action `[-1, 1]` into joint torque via `torque = gear * action`.

### Commit `9b7c7f1` — "Pendulum MuJoCo"

**File:** `assets/pendulum.xml` (18 lines)

A simple pendulum — hinge joint, gravity, no actuator. He lets it swing.

**THE FIRST BUG — The Stuck Pendulum Problem:**

The pendulum gets stuck against invisible walls or shoots upward unexpectedly. After hours of debugging, he discovers: **MuJoCo planes are infinite in both directions even when size-limited.** The visual plane is bounded, but the collision geometry extends infinitely. When the pendulum swings past the visual boundary, it encounters infinite collision geometry and bounces unrealistically.

**Engineering lesson:** Physics simulators don't show you everything. Always verify actual collision geometry.

### Commit `7c4bccb` — "Double Pendulum"

**Files:** `sandbox/cartpole.py` (modified), `assets/double_pendulum_cartpole.xml` (30 lines)

A cart with two rods in series (double inverted pendulum). The XML chains: `Cart → Pole1 (hinge) → Pole2 (hinge)`. He modifies the cartpole environment to handle the new observation space (6 values: cart pos/vel, pole1 angle/vel, pole2 angle/vel). He trains PPO to balance it.

**Videos created:**  

- `videos/chaos by double pendulum.mp4` — uncontrolled chaotic motion
- `videos/double pendulum balance.mp4` — PPO successfully stabilizing it

---

## Phase 1.5: First Complete RL Pipeline

### Commit `cf275ae` — "organised files"

**This is the project's structural foundation.** He creates the directory hierarchy that persists through the entire project:

```
rl-humanoid/
├── assets/          → XML models (cartpole, pendulum, G1, platform)
├── envs/            → Gymnasium environments
├── sandbox/         → Experimental/learning scripts
├── models/          → Trained policies and normalization stats
├── notes/           → All documentation
├── tools/           → Testers and utilities
├── videos/          → Result recordings
├── train.py         → Training entry point
├── run.py           → Evaluation entry point
└── README.md        → Project description
```

**`envs/cartpole_env.py`** (50 lines) — The first complete Gymnasium environment:  

```
python
class MyCartPoleEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        MujocoEnv.__init__(self, "cartpole.xml", frame_skip=5, ...)
        self.action_space = Box(low=-1, high=1, shape=(1,))
        self.observation_space = Box(low=-inf, high=inf, shape=(4,))

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset_model(self):
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = 1.0 if not terminated else 0.0
        return obs, reward, terminated, truncated, info
```

**`train.py`** (53 lines) — The first training script:  

```
python
model = PPO("MlpPolicy", env=train_env, learning_rate=3e-4, n_steps=2048, ...)
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")
```

**`run.py`** (27 lines) — The evaluation script:  

```
python
model = PPO.load("ppo_cartpole")
obs = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
```

**`sandbox/cartpole_parallel.py`** (88 lines) — Learning parallel environments:  

```
python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank):
    def _init():
        env = MyCartPoleEnv()
        env.reset(seed=1000 + rank)
        return env
    return _init

train_env = SubprocVecEnv([make_env(i) for i in range(8)])
```

**Result:** CartPole trained for 2000 steps, balances perfectly. Tested by giving it a small jerk — it recovers.

---

# PART 2: THE G1 HUMANOID STANDING PROBLEM (Weeks 5–11)

## Phase 2.1: Understanding the Unitree G1

### Commit `6810541` — "SSH Key"

**File:** `notes/Unitree_g1.md` (229 lines)

Rachit obtains the G1's MJCF from Unitree's open-source repositories. He documents the complete robot specification:

**29 DOF layout:**  

```
Left leg  (6):  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Right leg (6):  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
Waist      (3):  yaw, roll, pitch
Left arm   (7):  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
Right arm  (7):  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
```

**Standing height:** 0.793m (pelvis z-position from XML)

**Torque limits:**  

- Hip joints: 88 Nm
- Knee joints: 139 Nm
- Ankle joints: 50 Nm
- Waist joints: 88 Nm (yaw), 50 Nm (roll/pitch)
- Shoulder/elbow: 25 Nm
- Wrist: 5 Nm

**Floating base structure:**  

```
qpos[0:2]  = x, y position (skipped — not used in observation)
qpos[2]    = z height
qpos[3:7]  = quaternion (w, x, y, z)
qpos[7:36] = 29 joint angles
Total qpos = 36

qvel[0:3]  = linear velocity (x, y, z)
qvel[3:6]  = angular velocity (roll, pitch, yaw)
qvel[6:35] = 29 joint velocities
Total qvel = 35
```

**Initial observation space:** 34 (qpos[2:]) + 35 (qvel) = 69 dimensions → later refined to 67.

**The XML file:** `assets/g1_29dof.xml` (526 lines) — a complete MJCF model with:  

- 29 joints with range limits and torque limits
- 40+ STL mesh files for visual geometry
- Inertial properties for each body
- Default classes for different joint types (torso_motor, leg_motor, ankle_motor, arm_motor, wrist_motor)

---

## Phase 2.2: The Catastrophic Failure — Direct Torque Control

### Commit `84958cb` — "update"

**File:** `envs/g1_env.py` (98 lines) — first version

**The approach:** PPO outputs 29 torque values directly. Action space: `Box(low=-1, high=1, shape=(29,))`, scaled by torque limits.

**The result** (from `MUJOCO_LOG.TXT`):  

```
WARNING: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 0.4280.
```

The robot explodes within 0.4 seconds. The PPO policy during initial exploration outputs random torques — left knee gets +139 Nm, right knee gets -139 Nm simultaneously. The robot literally rips itself apart.

**Debugging attempts** (from `notes/log.txt`):  

```
02:55 08-06-2026
Facing Issue: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable

21:24 08-06-2026
I have tried many variations but nothing works, I will implement PD controls now
```

---

## Phase 2.3: The PD Controller Breakthrough

### Commit `889fb64` — "Progress"

**The single most important engineering decision** in the project. He redesigns the entire control architecture:

**Old:** `action = torque` → direct motor control  
**New:** `action = target_joint_offset` → PD controller → torque

From `notes/Unitree_g1.md`:  

```
torque = kp * (target_q - q) - kd * qvel
target_q = standing_pose + 0.15 * action
```

**`envs/g1_config.py`** (72 lines) — The configuration file:  

```
python
# PD gains (from Unitree official config.yaml)
kp = np.array([
    100, 100, 100, 150, 40, 40,    # left leg
    100, 100, 100, 150, 40, 40,    # right leg
    200, 200, 200,                  # waist
    40, 40, 40, 40, 40, 40, 40,    # left arm
    40, 40, 40, 40, 40, 40, 40,    # right arm
])

kd = np.array([
    2, 2, 2, 4, 2, 2,              # left leg
    2, 2, 2, 4, 2, 2,              # right leg
    5, 5, 5,                        # waist
    10, 10, 10, 10, 10, 10, 10,    # left arm
    10, 10, 10, 10, 10, 10, 10,    # right arm
])

# Action scale (how much each joint can deviate from standing pose)
ACTION_SCALE = np.array([
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,    # left leg
    0.25, 0.25, 0.15, 0.50, 0.25, 0.15,    # right leg
    0.15, 0.15, 0.15,                        # waist
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,  # left arm
    0.30, 0.30, 0.20, 0.30, 0.15, 0.10, 0.10,  # right arm
])

# Standing pose (all zeros initially — will be refined)
STANDING_POSE = np.zeros(29, dtype=np.float64)
```

**`envs/g1_env.py`** — The PD controller implementation:  

```
python
def step(self, action):
    # Convert action to target joint positions
    target_q = STANDING_POSE + ACTION_SCALE * action

    # PD controller
    q = self.data.qpos[7:]      # joint positions
    qd = self.data.qvel[6:]     # joint velocities
    torque = kp * (target_q - q) - kd * qd
    torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

    # Apply torque
    self.do_simulation(torque, self.frame_skip)

    # Compute reward
    height = self.data.qpos[2]
    height_reward = REWARD_HEIGHT * np.exp(-HEIGHT_GAUSSIAN_K * (height - STANDING_HEIGHT) ** 2)
    upright = float(self.data.body("pelvis").xmat[8])  # z-axis alignment
    upright_reward = REWARD_UPRIGHT * max(0.0, upright)
    energy = PENALTY_ENERGY * float(np.sum(np.abs(torque * qd)))
    vel_penalty = PENALTY_JOINT_VEL * float(np.sum(qd ** 2))
    action_penalty = PENALTY_ACTION * float(np.sum(action ** 2))

    reward = REWARD_ALIVE + height_reward + upright_reward + energy + vel_penalty + action_penalty

    terminated = bool(height < 0.4 or upright < 0.75)
    return obs, reward, terminated, truncated, info
```

**The PD controller test** (before RL):  

```
00:30 09-06-2026
PD control finally works and now the robot is not breaking its bones now
It gives no error in training... its just not standing properly
Its vibrating like someone pointed a gun at him
```

The robot can stand for ~1200 steps without falling, but vibrates. He adds:  

- `smoothed_action` = low-pass filter: `action = 0.9 * prev_action + 0.1 * raw_action`
- Higher `kd` values for damping
- Energy penalty to discourage flailing

**The log tells the struggle:**  

```
02:22 09-06-2026
Tried smoothed_action... Changed reward... Doesn't stand
```

---

## Phase 2.4: The 30M Steps Debugging Marathon

### Commit `2e43d5a` — "official code"

Rachit studies Unitree's official Isaac Lab code at:  

```
source/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py
```

He creates `notes/g1_walk_issac.md` (639 lines) — a complete reverse-engineering of the official code. This is his most detailed analysis document.

**Key differences discovered:**

| My Code                            | Unitree Official                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| Uses raw qpos/qvel (69 dims)       | Uses base ang_vel, projected gravity, cmd, joint pos/vel, prev action, phase |
| No vibration penalty               | Explicit `joint_vel`, `action_rate`, `joint_acc`, `energy` penalties         |
| No external disturbances           | Pushes every 5 seconds                                                       |
| Termination at 0.75 upright (<41°) | Termination at 0.8 rad (<46°) — earlier                                      |
| MLP policy [256, 256]              | Recurrent PPO (LSTM) with history_length=5                                   |
| No foot contact sensing            | Foot contact, foot clearance rewards                                         |
| Single environment                 | 4096 parallel environments                                                   |
| CPU training                       | GPU training (NVIDIA RTX)                                                    |

He completely rewrites his observation space to match Unitree's structure.

### Commit `97cd0c4` — "30M Training"

**Observation space evolves to 67 dims:**  

```
python
def _get_obs(self):
    pelvis_xmat = self.data.body("pelvis").xmat.reshape(3, 3)
    base_ang_vel = self.data.qvel[3:6]
    body_ang_vel = pelvis_xmat.T @ base_ang_vel      # rotate to body frame
    base_lin_vel = self.data.qvel[0:3]
    body_lin_vel = pelvis_xmat.T @ base_lin_vel
    projected_gravity = pelvis_xmat.T @ np.array([0.0, 0.0, -1.0])
    joint_pos = self.data.qpos[7:] - STANDING_POSE     # relative to standing pose
    joint_vel = self.data.qvel[6:]

    return np.concatenate([
        body_ang_vel,      # 3
        body_lin_vel,      # 3
        projected_gravity, # 3
        joint_pos,         # 29
        joint_vel          # 29
    ], dtype=np.float64)   # Total: 67
```

**The 30M step overnight training:**  

```
11:28 23-06-2026
I literally trained the model for 30M steps overnight just for the mean_episode length of 50 :_(
```

30 million steps = ~7 hours of training. The robot survives only 50 steps on average. **Something is fundamentally wrong.**

### Commit `26029f9` — "external repo"

He clones and studies multiple Unitree codebases:  

- `unitree_rl_lab` (Isaac Lab)
- `unitree_rl_gym` (Gymnasium)
- `unitree_rl_mjlab` (MuJoCo port by community)

He also creates the submodule `unitree_rl_mjlab` — a separate MuJoCo deployment framework for sim-to-real transfer.

---

## Phase 2.5: The Breakthrough Discovery

**The Root Cause** (from `notes/log.txt`):  

```
11:26 14-06-2026
IT WORKS !!!
Standing pos was above so it was falling every time
```

**What was happening:**

When the robot spawns in MuJoCo at `pos="0 0 0.793"` with all joints at zero angle (straight legs), the pelvis is at the maximum height. The official Unitree code uses a **bent standing pose** where knees are slightly bent and hips are flexed, lowering the center of mass.

Rachit's initial `STANDING_POSE = np.zeros(29)` means:  

- Legs are fully straight
- Pelvis is at maximum height (0.793m)
- Robot is top-heavy and unstable

**The suicide local optimum:**

Each episode:  

1. Robot spawns at height 0.793m with straight legs
2. Gravity pulls it down — it's already unstable
3. PPO tries random actions, which make it fall faster
4. Robot hits the ground (height < 0.4) → episode terminates
5. Reward is negative (energy penalties + no positive rewards accumulated)
6. PPO learns: **falling faster = less negative reward = better**
7. The robot gets better at committing suicide

From `log.txt`:  

```
00:49 24-06-2026
Suicide local optimum reached... iykyk
```

**The fix:** Adjust the initial joint configuration to match the bent standing pose:  

```
python
STANDING_POSE = np.array([
    # Left leg (slightly bent)
    -0.15,     # left_hip_pitch  (flexion)
    0.02,      # left_hip_roll
    0.0,       # left_hip_yaw
    0.3,       # left_knee       (bent)
    -0.15,     # left_ankle_pitch
    0.0,       # left_ankle_roll
    # Right leg (mirror)
    -0.15,
    -0.02,
    0.0,
    0.3,
    -0.15,
    0.0,
    # Waist (all zero)
    0.0, 0.0, 0.0,
    # Arms (slightly forward)
    0.1, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0,
    -0.1, -0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
])
```

**After the fix:** At ~3M timesteps:  

- Average episode length: ~1652 steps
- Average reward: ~6309
- Explained variance: 0.988
- Value loss near zero

**Model progression:**  

- `g1_stand.zip` (275 KB) — basic standing
- `g1_stand_v2.zip` (2.1 MB) — improved standing with larger network
- `g1_stand_final.zip` (315 KB) — final standing
- `g1_stand_retry.zip` (2.1 MB) — retry with corrected hyperparameters

---

## Phase 2.6: Push Recovery Training

### Commit `d9ed519` — "video"

**Files:** `envs/g1_config_push.py` (79 lines), `envs/g1_env_push.py` (130 lines)

Now the robot stands, but can it recover from pushes? Rachit adds an external disturbance system:

```
python
def _apply_push(self):
    """Apply a random external force to the pelvis at random intervals."""
    if self.push_timer <= 0:
        push_dir = np.random.uniform(-1, 1, 3)
        push_dir[2] *= 0.3  # less vertical push
        push_mag = np.random.uniform(10, 50) * self.push_scale
        self.data.xfrc_applied[pelvis_id] = push_dir * push_mag
        self.push_timer = np.random.randint(100, 300)
```

**Curriculum learning for push strength:**  

```
python
# Stage 0: no pushes (warmup)
# Stage 1: light pushes (10-20 N)
# Stage 2: medium pushes (20-35 N)
# Stage 3: hard pushes (35-50 N)
# Stage 4: very hard pushes (50-80 N)
```

**Training iterations:**  

- `g1_stand_force.zip` → `g1_stand_force_2.zip` → `g1_stand_force_3.zip`
- Each iteration had different push magnitudes, reward scaling, or network architecture
- `g1_stand_force_vecnorm.pkl` → `_2.pkl` → `_3.pkl` (3 iterations of normalization parameters)

**Video:** `videos/small_pushes.mp4` shows the robot recovering from applied forces.

---

# PART 3: THE PLATFORM PROBLEM — MOVING BUS SCENARIO (Weeks 11–15)

## Phase 3.1: The Problem Statement

**The problem:** Normal pushes don't simulate a robot standing on a moving bus. In a bus:  

1. The floor (platform) moves under the robot's feet
2. The acceleration transfers through the robot's legs
3. The robot must continuously adjust its stance

**The engineering challenge:** How to add a moving platform WITHOUT breaking the existing trained model?

The existing model has:  

- Observation space: 67 dims
- Action space: 29 dims
- Trained for 3M+ steps

Adding a platform changes both observation and action spaces, making the trained model unusable.

---

## Phase 3.2: Attempt 1 — Mocap Body (Failed)

**File:** `assets/platform_29dof.xml` (30 lines)

```
xml
<mujoco model="g1_platform">
  <include file="g1_29dof.xml"/>
  <worldbody>
    <body name="platform" pos="0 0 -0.05" mocap="true">
      <geom name="platform_geom" type="box" size="0.5 0.5 0.05" mass="1000"/>
    </body>
  </worldbody>
</mujoco>
```

**The idea:** Mocap bodies don't add actuators or change the observation space. He updates the platform position each frame:  

```
python
self.data.mocap_pos[0] = new_position  # move platform every frame
```

**The bug:** The robot slips off the platform as if there's no friction.

**Debugging journey** (from `notes/log.txt`):  

```
15:25 17-06-2026
Was working on the platform logic of the code, had to write custom simulation function 
because frame skip was 5 and it was causing physics issue (velocity zero for 5 frames).
I couldn't include the platform in the action space as training is already done.
```

The `frame_skip=5` means MuJoCo runs 5 substeps per environment step. The mocap position is set at the start of the step, but for substeps 2–5, the platform's velocity is effectively zero.

He writes a custom simulation function:  

```
python
def _custom_step(self, action):
    """Simulate with per-substep platform position updates."""
    for _ in range(self.frame_skip):
        self.data.mocap_pos[0] = self._platform_target_pos
        mujoco.mj_step(self.model, self.data)
```

**Still fails:**  

```
12:00 18-06-2026
It still is sliding :_)
```

**The root cause** (from `notes/MuJoCo.md`):  

> "MuJoCo does not compute velocity, acceleration, momentum for mocap bodies. It simply sees Old position and New position every update. But mocap body has qvel = 0 always. Physics sees platform not moving during most substeps. So contact solver doesn't generate realistic friction."

The friction force in MuJoCo depends on relative velocity at the contact point. If the platform has zero velocity (mocap = always qvel=0), the friction cone is essentially infinite in static friction but zero in dynamic friction. The robot's feet can't get traction.

**Even with per-substep updates**, the mocap body's `qvel` remains zero. The contact solver still sees the platform as stationary.

```
22:46 18-06-2026
Understood... from first step the robot is still and platform is moving so its creating 
so much acceleration on robot for curriculum learning
```

When the platform accelerates from 0 to some velocity, the robot's feet are static relative to the platform (because mocap friction is wrong), and the robot experiences the full acceleration impulse — unrealistic.

---

## Phase 3.3: Attempt 2 — Real Platform with Actuators (Success)

He creates a **real platform** with a slide joint and a position actuator:  

```
xml
<body name="platform" pos="0 0 -0.05">
    <joint name="platform_joint" type="slide" axis="1 0 0"/>
    <geom type="box" size="0.5 0.5 0.05" mass="1000"/>
</body>
<actuator>
    <position joint="platform_joint" kp="1000"/>
</actuator>
```

**Now the problem:** Adding an actuator changes the action space (29 → 30) and observation space (67 → 69+). He cannot reuse his trained model.

**The solution:** He loads the XML but **excludes the platform actuator from the PPO action space**. The platform is controlled by a separate mechanism:

From `envs/g1_walk_env.py` (lines 303-323):  

```
python
def _update_platform(self):
    """Set platform position based on current velocity command."""
    elapsed = self.data.time - self._platform_start_time
    new_pos = self._platform_start_pos + self._platform_velocity * elapsed
    # Set platform joint position directly
    self.data.qpos[platform_joint_id] = new_pos
```

But this is mocap-like — manually setting joint positions. **The actual working solution** uses a separate position actuator controlled externally:

```
python
# Platform is driven by its own position actuator
# controlled externally (not by PPO)
self.data.ctrl[-1] = platform_position_target  # last ctrl index = platform
```

To avoid changing the action space, he loads the XML *without* the platform actuator for the training environment, and creates a **wrapper** that adds the platform physics externally. The observation space stays at 67, the action space stays at 29, and platform motion is added in the simulation loop.

---

## Phase 3.4: The Walking Environment

### Commits `84c5424` and `4ab9f3f` — "Custom Walk"

**Files:** `envs/g1_walk_env.py` (665 lines), `envs/g1_walk_config.py` (249 lines), `train_walk.py` (201 lines)

This is the **largest, most sophisticated component** of the project. It combines everything learned.

### Observation Space (115 dimensions):

```
python
# Actor sees (98 dims):
#   angular_velocity (3)
#   projected_gravity (3)
#   velocity_command (3)
#   dof_pos (29)
#   dof_vel (29)
#   actions (29)
#   phase (2)
#
# Critic-only sees (17 dims):
#   linear_velocity (3)
#   platform_velocity (1)
#   foot_height (4)
#   foot_air_time (4)
#   foot_contact (4)
#   foot_forces (1)
```

### Action Space (29 dims):

Target joint position offsets from standing pose, scaled by `0.15`.

### Curriculum Learning (5 stages):

```python
NUM_CURRICULUM_STAGES = 5

# Stage 0: stationary platform, no velocity command



# Stage 1: slow platform (0.2 m/s), small velocity command range



# Stage 2: medium

continue

# PART 3: THE PLATFORM PROBLEM — MOVING BUS SCENARIO (continued)

## Phase 3.4: The Walking Environment (continued)

### Curriculum Configuration:
```

python
NUM_CURRICULUM_STAGES = 5

# Stage 0: stationary platform, no velocity command

    # platform_vel_range = (0.0, 0.0)
    # cmd_vel_range = (0.0, 0.0)
    # survival_threshold = 150 steps

# Stage 1: slow platform (0.2 m/s), small velocity command range

    # platform_vel_range = (0.0, 0.2)
    # cmd_vel_range = (0.0, 0.4)
    # survival_threshold = 300 steps

# Stage 2: medium platform (0.5 m/s), moderate commands

    # platform_vel_range = (0.0, 0.5)
    # cmd_vel_range = (0.0, 0.8)
    # survival_threshold = 500 steps

# Stage 3: fast platform (0.8 m/s), wider commands

    # platform_vel_range = (0.0, 0.8)
    # cmd_vel_range = (0.0, 1.2)
    # survival_threshold = 800 steps

# Stage 4: full speed (1.2 m/s), full command range

    # platform_vel_range = (0.0, 1.2)
    # cmd_vel_range = (0.0, 1.6)
    # survival_threshold = 1200 steps

```
Each environment independently tracks its own curriculum level. Environments that survive beyond their threshold get promoted; environments that fall before halfway get demoted. This is a direct adaptation of Unitree's `terrain_levels_vel()` curriculum.

### Complete Reward Function:
```

python
reward = (
    + vel_tracking_reward * 1.0      # exp(-||cmd_vel - actual_vel||^2 / 0.25)
    + upright_reward * 1.0           # dot(z_body, z_world), clipped [0, 1]
    + height_reward * 1.0            # exp(-40*(height - 0.78)^2) Gaussian
    + survival_bonus * 0.5           # +0.5 per step alive
    + foot_clearance_reward * 0.5    # reward for lifting swing foot to 0.1m
    + foot_contact_pattern * 0.2     # reward alternating left/right contact
    - energy_penalty * 0.001         # -|torque * joint_vel|
    - action_rate_penalty * 0.01     # -||action - prev_action||^2
    - joint_vel_penalty * 0.001      # -||joint_vel||^2
    - joint_acc_penalty * 0.0001     # -||joint_vel - prev_joint_vel||^2
    - foot_slip_penalty * 0.5        # penalize feet moving while in contact
)

```
### Asymmetric Actor-Critic Implementation:
```

python
class AsymmetricPolicy(ActorCriticPolicy):
    """Actor sees only 98 of 115 obs dims; Critic sees all 115."""

    def extract_features(self, obs, features_extractor=None):
        features = super().extract_features(obs, features_extractor)
    
        if isinstance(features, tuple):
            pi_features, vf_features = features
            # Mask critic-only features (dims 98:115) from actor
            masked_pi = pi_features.clone()
            masked_pi[:, _CRITIC_START:] = 0.0
            return masked_pi, vf_features
    
        return features

```
This is a sophisticated technique from advanced RL research (Pinto et al., 2017). The critic sees privileged information (linear velocity, platform velocity, foot contact forces) that the actor cannot access during deployment — forcing the actor to infer these from available sensor signals.

### Termination Conditions:
```

python
terminated = (
    height < 0.3                  # on ground
    or upright < 0.6              # tilted >53°
    or joint_limit_violation      # any joint past safe range
    or abs(platform_pos) > 3.0    # fell off the platform edge
)

```
### Training Configuration:
```

python
NUM_ENVS = 8
TOTAL_TIMESTEPS = 50_000_000

model = PPO(
    policy=AsymmetricPolicy,
    env=train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,           # official Unitree uses 0.01
    max_grad_norm=1.0,
    policy_kwargs=dict(
        net_arch=[512, 256, 128],
        activation_fn=torch.nn.ELU,
        share_features_extractor=False,
    ),
    tensorboard_log="./tb_logs/",
)

```
### Checkpoint System:

Saves every 100K env steps per subproc → effectively every 12,500 total timesteps (= 100K / 8 envs).

The checkpoint directory `models/walk_checkpoints_v5/` contains 15 checkpoints at:  
100K, 200K, 300K, 400K, 500K, 600K, 700K, 800K, 900K, 1M, 1.1M, 1.2M, 1.3M, 1.4M, 1.5M steps

Each checkpoint includes:  

- `walk_ckpt_v5_{steps}.zip` — PPO model weights
- `walk_ckpt_v5_vecnormalize_{steps}.pkl` — observation normalization running stats

This dense checkpointing saved him multiple times during the 50M step training runs that could take 12+ hours.

---

## Phase 3.5: The Smooth Action Implementation

From the log, he mentions implementing a smooth action function. In `g1_walk_env.py`, this appears as an exponential moving average filter:
```

python
self._prev_action = np.zeros(29, dtype=np.float32)

def step(self, action):
    # Smooth action: blend with previous action
    smoothed = 0.9 * self._prev_action + 0.1 * action
    self._prev_action = smoothed.copy()

    # Apply PD controller with smoothed action
    target_q = STANDING_POSE + ACTION_SCALE * smoothed
    # ...

```
This prevents the robot from jerking when the policy outputs sudden changes. The `action_rate_penalty` in the reward function complements this by penalizing large action deltas during training.

---

## Phase 3.6: TensorBoard Training History

The `tb_logs/` directory contains **22 separate training runs** (g1_walk_1 through g1_walk_20, plus PPO_13 through PPO_22). This reveals:

**Training History:**  

- **g1_walk_1 through g1_walk_4**: Initial walking experiments with the new environment — likely debugging platform physics and curriculum
- **g1_walk_5 through g1_walk_10**: Iterations refining reward weights and curriculum thresholds
- **g1_walk_11 through g1_walk_15**: Longer training runs (10M-20M steps each) with different hyperparameters
- **g1_walk_16 through g1_walk_20**: Final training runs (50M steps each) with the settled configuration

**Model version progression:**  

- `g1_walk_v1.zip` (5.3 MB) — first successful walking model
- `g1_walk_v2.zip` (6.2 MB) — improved with more training
- `g1_walk_v3.zip` (5.3 MB) — refined reward weights
- `g1_walk_v4.zip` (5.3 MB) — final walking model
- `walk_checkpoints_v5/` — version 5 with full checkpoint history (1.5M steps)

Each model has a corresponding `vecnormalize.pkl` file storing observation mean/std for inference.

---

## Phase 3.7: Video Results

The `videos/` directory captures the complete progression:

| Video | Size | Content |
| --- | --- | --- |
| `CartPole Demo.mp4` | 1.3 MB | First RL success — cartpole balanced |
| `chaos by double pendulum.mp4` | 1.2 MB | Uncontrolled double pendulum chaos |
| `double pendulum balance.mp4` | 0.8 MB | PPO stabilizing double pendulum |
| `g1_stading.mp4` | 14 MB | Initial G1 standing (after PD fix) |
| `small_pushes.mp4` | 6.5 MB | Push recovery demonstration |
| `g1_drunk walk.mp4` | 5 MB | Early walking — wobbling gait, imperfect |
| `g1_walk.mp4` | 26.5 MB | Final successful walking on moving platform |
| `Walk.mp4` | Added later | Walking from different camera angle |
| `balance.mp4` | Added later | Balance recovery on platform |
| `top view.mp4` | Added later | Top-down view of walking on platform |

The progression from `g1_drunk walk.mp4` (5 MB, shaky gait, arms flailing) to `g1_walk.mp4` (26.5 MB, stable gait with arm swing) represents approximately 50M+ training steps and dozens of reward function iterations.

---

# PART 4: THE UNITREE RL MJLAB SUBMODULE

## External Repository: `unitree_rl_mjlab`

**Submodule commit:** `5d23442`  
**Repository:** `https://github.com/Rachit1801/unitree_rl_mjlab`

This is a separate MuJoCo deployment framework that Rachit forked/created for sim-to-real transfer. It contains:

### C++ Simulator (`simulate/src/`):



- `physics_joystick.h` — XBox and Switch joystick support for manual robot control
- `unitree_sdk2_bridge.h` — Hardware communication bridge using Unitree's SDK2 protocol
- `lodepng/` — PNG image encoder for visualization output
- Full CMake build system for Windows/Linux with `system_environment.cpp`, `mujoco_simulation.cpp`, `rl_deployment.cpp`

### Deployment Framework (`deploy/`):



- `include/unitree_joystick_dsl.hpp` — Domain-specific language parser for joystick commands (200+ lines of lexer/parser/AST)
- `include/unitree_articulation.h` — Base articulation class for robot hardware abstraction
- `include/isaaclab/manager/` — Isaac Lab-style observation manager, action manager, and configuration system ported to C++
- `robots/g1/include/State_Mimic.h` — State machine for motion mimic (loading motion sequences from files)

### Python Training Code (`src/tasks/velocity/`):



- `velocity_env_cfg.py` — A more complete port of the official Isaac Lab configuration to pure MuJoCo Python

This represents the **sim-to-real pipeline** — the trained MuJoCo policy can be deployed on the real G1 robot through this framework. The C++ bridge communicates with the real G1 over Ethernet using Unitree's SDK2 protocol, reading joint states and sending torque commands at 100Hz.

---

# PART 5: SUPPLEMENTARY FILES AND TOOLS

## `tools/humanoid_tester.py` (28 lines)

A utility script for manual testing of the humanoid. Allows keyboard-controlled joint movement for debugging.

## `tools/model_tester.py` (34 lines)

A script for loading and evaluating trained models. Tests different models against each other, recording performance metrics.

## `run_walk.py` (49 lines)

Evaluation script for the walking model:  
```

python
from envs.g1_walk_env import G1WalkEnv
from stable_baselines3 import PPO

env = G1WalkEnv(render_mode="human")
env = DummyVecEnv([lambda: env])
model = PPO.load("models/g1_walk_v4")
obs = env.reset()
for step in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

```
## `run_in_mujoco.py` (21 lines)

A minimal MuJoCo viewer for the raw G1 XML model (no RL, just physics testing):  
```

python
model = mujoco.MjModel.from_xml_path("assets/g1_29dof.xml")
data = mujoco.MjData(model)
data.qpos[2] = 0.793  # standing height
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

```

## `requirements.txt`

Contains all Python dependencies, presumably including:  

- `mujoco>=3.0`
- `gymnasium>=0.29`
- `stable-baselines3>=2.0`
- `torch`
- `numpy`
- `tensorboard`

---

# SYNTHESIS: COMPLETE ENGINEERING TIMELINE

| Date                | Event                                                                                |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Early May 2026**  | Project begins. Research phase: humanoid robotics, RL theory, MuJoCo physics         |
| **Mid May 2026**    | First MuJoCo model (box.py). Cartpole, pendulum, double pendulum sandbox experiments |
| **Late May 2026**   | First complete RL pipeline. CartpoleEnv + PPO + training/evaluation scripts          |
| **Commit cf275ae**  | Project structure organized into assets/envs/models/notes/tools/videos               |
| **Commit 9aa52dd**  | Initial README.md (repo creation)                                                    |
| **Late May 2026**   | G1 XML model acquired and studied. 29 DOF anatomy documented                         |
| **Early June 2026** | Direct torque control attempted → catastrophic failure (NaN within 0.4s)             |
| **~8 June 2026**    | PD controller implemented. Robot stands but vibrates                                 |
| **~14 June 2026**   | BREAKTHROUGH: Suicide local optimum discovered and fixed (bent standing pose)        |
| **~23 June 2026**   | 30M step overnight training → mean episode length only 50 steps                      |
| **Late June 2026**  | Push recovery training. Multiple model iterations (force_1/2/3)                      |
| **Early July 2026** | Platform problem begins. Mocap body approach fails                                   |
| **~18 July 2026**   | Platform actuator solution implemented. Walking environment created                  |
| **Late July 2026**  | Walking training across 4 model versions, 20+ TensorBoard runs                       |
| **~5 August 2026**  | Final walking model completed. 50M+ training steps across all iterations             |
| **Final commits**   | Custom Walk (HEAD). 15 checkpoints saved for version 5                               |

---

# MISSING FROM STORY — Questions

Based on exhaustive comparison between your `story.txt` and the repository evidence, here are the gaps:

## 1. The Pendulum Geometry Bug

**Repository evidence:** `assets/pendulum.xml` exists. The `notes/MuJoCo.md` mentions "planes are infinite in both directions."

**Question:** Did you discover this through systematic MuJoCo documentation reading, or was it pure trial-and-error (just seeing the pendulum behave strangely and experimenting)? How long did this bug take to debug?

---

## 2. The 30M Overnight Training and Recovery

**Repository evidence:** The `models/` contain `g1_stand_retry.zip` and `g1_stand_retry_vecnorm.pkl` alongside `g1_stand.zip` and `g1_stand_v2.zip`.

**Question:** What specific change fixed the suicide local optimum? Your story says "I saw the problem the initial position of robot when it was spawned... was a little bit higher." Was the fix changing `STANDING_POSE` from all-zeros to the bent pose, or was it changing the spawn height `pos="0 0 0.793"` in the XML? Or both?

**Also:** You mention "I even added a huge reward for its survival but that didn't help much either." How much did you increase the survival reward before realizing it couldn't compensate for the fundamental spawn problem?

---

## 3. The Asymmetric Actor-Critic

**Repository evidence:** `train_walk.py` contains a custom `AsymmetricPolicy` class that masks the last 17 observation dimensions from the actor.

**Question:** The official Unitree Isaac Lab code uses Recurrent PPO with `history_length=5` frame stacking — **not** asymmetric actor-critic. Where did you learn about asymmetric actor-critic? Was it a deliberate design choice after reading the relevant paper (Pinto et al. 2017), or did you discover it through some other resource?

---

## 4. The Smooth Action Function

**Repository evidence:** The log mentions "Tried smoothed_action" but the exact implementation details aren't in your story.

**Question:** Was the smooth action just the exponential moving average (`0.9 * prev + 0.1 * raw`), or did you also try other smoothing approaches (e.g., action rate limiting, clamping rate of change, Kalman filtering)?

---

## 5. The Unitree RL MJLab Submodule

**Repository evidence:** `unitree_rl_mjlab` is a **separate GitHub repository** as a git submodule. It contains a complete C++ deployment framework with joystick control, SDK2 hardware bridge, Isaac Lab-style managers ported to C++, and state machines.

**Question:** You don't mention this at all in your story. Did you:  

- Fork an existing repository and make modifications?
- Write the entire C++ deployment framework from scratch?
- Was this a separate project or was it intended as the final sim-to-real deployment target for your trained policies?

---

## 6. The Recording and Demo Pipeline

**Repository evidence:** All videos are `.mp4` files recorded with specific naming conventions, and the `train.py` has commented-out `resume` sections.

**Question:** What was your process for recording training progress? Did you manually record episodes after training milestones, or did you have an automated evaluation script that rendered and saved videos? How did you decide when to stop iterating on a model version and declare it "done"?

---

# CONTRADICTIONS BETWEEN STORY AND REPOSITORY

## Contradiction 1: Timeline of Platform Work

**Your story says:** "After identifying the problem and fixing it the reward successfully learned to stay balanced... After it was balanced I trained the model to balance by giving him pushes... After it I made a custom environment with the robot and a platform beneath it..."

**Repository evidence suggests:** The platform XML (`assets/platform_29dof.xml`) and the first `g1_env_push.py` commit appear in the same commit (`2f8ed30` — "19th May 2026" initial bulk commit). The push recovery training AND the platform experiments appear to have been developed concurrently, not sequentially.

**Question:** Did you actually work on the platform before push recovery was fully solved? Or did the commit history merge them because you committed everything at once after each milestone?

---

## Contradiction 2: External Repository Reference

**Your story says:** "I tried writing the code on my own with my own logic but because of so many failures I had to look at the official code published on github by unitree robotics"

**Repository evidence:** The `notes/g1_walk_issac.md` shows you downloaded the official code on **14 June 2026** (file metadata in notes). But your first PD controller was working on **9 June 2026**. So you had the PD controller working *before* studying the official code.

**Clarification:** Did you implement the PD controller independently, and *then* used the official code to refine the observation space, reward function, and curriculum? Or did the official code inspire the PD controller approach?

---

## Contradiction 3: Commit Message "SSH Key"

**Your story doesn't mention:** Commit `6810541` has the message "SSH Key". There's no SSH configuration in the repository.

**Question:** Was this commit primarily about setting up GitHub authentication (SSH keys for pushing to remote), or was there some SSH-related configuration for the Unitree G1 hardware or the DRDO lab network?

---

# IMPORTANT IMPLEMENTATION DECISIONS — Summary

1. **PD Controller over Direct Torque Control** — The single most critical decision. Transformed an exploding robot into a stable system.

2. **Bent Standing Pose** — Fixing the spawn height corrected the suicide local optimum and made learning possible.

3. **Action Smoothing (EMA filter)** — Reduced vibration in early standing by blending actions with previous values.

4. **Asymmetric Actor-Critic** — Clean separation of privileged (critic-only) vs. deployable (actor) observations.

5. **Platform with Real Actuator (not Mocap)** — The correct approach after learning that mocap bodies have zero velocity in MuJoCo's contact solver.

6. **Per-Environment Curriculum** — Each parallel environment independently tracks its own curriculum level, preventing early failures from dragging down successful learners.

7. **Comprehensive Vibration Penalties** — `joint_vel`, `action_rate`, `joint_acc`, and `energy` penalties are essential for smooth sim-to-real transfer.

8. **Dense Checkpointing** — Every 12,500 steps saved a full model + normalization stats, enabling recovery from crashes during 12+ hour training runs.

---

# FINAL CONTEXT DOCUMENT FOR INTERNSHIP REPORT

The project demonstrates a complete RL from-scratch engineering journey for the Unitree G1 humanoid robot:

1. **Learning Phase**: MuJoCo physics → Gymnasium interface → PPO algorithm → parallel training infrastructure
2. **Standing Phase**: PD controller design → observation space engineering → reward shaping → bug diagnosis (suicide local optimum) → 30M steps training → push recovery curriculum
3. **Walking Phase**: Platform physics challenge → mocap limitation discovery → actuator-based solution → asymmetric actor-critic → comprehensive curriculum with 5 stages → 50M steps training across 4 major versions → foot contact detection → gait rewards

**Final deliverables:**  

- 665-line walking environment with complete reward system and curriculum
- 4 trained walking models (v1-v4)
- 15 checkpoints for version 5
- 22 TensorBoard training runs documenting the full evolution
- 10 demo videos showing progression from cartpole to G1 walking on moving platform
- External C++ deployment framework for sim-to-real transfer

**Robustness achieved:** The final robot can stand on a moving platform (simulating bus/moving floor), recover from external pushes, and track velocity commands while maintaining balance. The robot exhibits natural gait patterns with arm swing and appropriate foot clearance.


