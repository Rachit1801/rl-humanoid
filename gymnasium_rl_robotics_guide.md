# 🤖 Gymnasium for Robotics: A Complete Beginner-to-Practical Guide

> **Target Audience:** Absolute beginners to RL, learning Gymnasium for humanoid robotics and simulation environments like MuJoCo and Isaac Lab.
> **Teaching Philosophy:** Why → What → How. Intuition first, math second. Robotics analogies throughout.

---

## Table of Contents

1. [What Gymnasium Actually Is](#1-what-gymnasium-actually-is)
2. [The Core RL Loop](#2-the-core-rl-loop)
3. [Gymnasium API Deep Dive](#3-gymnasium-api-deep-dive)
4. [Understanding `env.step()` Deeply](#4-understanding-envstep-deeply)
5. [Writing Your First Gymnasium Program](#5-writing-your-first-gymnasium-program)
6. [Gymnasium + MuJoCo](#6-gymnasium--mujoco)
7. [Environment Design Internals](#7-environment-design-internals)
8. [Building a Custom Environment](#8-building-a-custom-environment)
9. [Gymnasium + PPO (Stable-Baselines3)](#9-gymnasium--ppo-stable-baselines3)
10. ****[The Robotics Perspective: From Sim to Real](#10-the-robotics-perspective-from-sim-to-real)

---

## 1. What Gymnasium Actually Is

### 1.1 The Problem It Solves

Imagine you are training a robot dog to walk. You write a smart learning algorithm. Now you want to also train a humanoid arm to pick up objects, and later a drone to navigate a maze.

**Without Gymnasium**, you would face this nightmare:

- Every environment (robot simulator, game, grid world) has a completely different API.
- Your algorithm code gets tangled with environment code.
- You can't swap environments without rewriting your entire training loop.
- You can't compare algorithms fairly because environments behave differently.

**Gymnasium solves this** by defining a single, universal contract:

> *"Every environment, no matter how complex, will expose the exact same interface."*

Think of it like a **USB standard for RL environments**. Your phone, keyboard, and hard drive are all completely different devices, but they all plug into USB the same way. Gymnasium is the USB standard for RL.

---

### 1.2 The Key Players (and How They Relate)

Before writing a single line of code, you must understand the **ecosystem map**. These five things are often confused:

```
┌─────────────────────────────────────────────────────────────────┐
│                        THE RL ECOSYSTEM                         │
│                                                                 │
│  ┌──────────────┐       talks to       ┌──────────────────┐     │
│  │  RL Algorithm│ ───────────────────► │   Environment    │     │
│  │  (PPO, SAC,  │ ◄─────────────────── │   (Gymnasium)    │     │
│  │   TD3, etc.) │    obs/reward/done   │                  │     │
│  └──────────────┘                      └────────┬─────────┘     │
│                                                 │               │
│                                         uses internally         │
│                                                 │               │
│                                        ┌────────▼─────────┐     │
│                                        │    Simulator     │     │
│                                        │  (MuJoCo /       │     │
│                                        │   Isaac Lab /    │     │
│                                        │   PyBullet)      │     │
│                                        └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

Now let's define each player clearly:

---

#### 🧠 RL Algorithm

The **brain** of your system. It decides what action to take, learns from rewards, and improves over time. Examples: PPO, SAC, TD3, DDPG.

*Robotics analogy:* The RL algorithm is like a physical therapist teaching a stroke patient to walk again — it observes the patient, suggests actions (move left leg forward), and adjusts based on what worked.

---

#### 🌍 Environment

The **world** the agent lives in. It receives actions, simulates what happens next, and returns observations + rewards. Gymnasium defines the interface for all environments.

*Robotics analogy:* The environment is the physical therapy room — it has gravity, obstacles, a floor. The patient (agent) interacts with this world.

---

#### ⚙️ Simulator

The **physics engine** running underneath the environment. It actually computes how joints move, how forces interact, how collisions happen.

*Robotics analogy:* The simulator is the laws of physics themselves — gravity, friction, inertia. The environment uses these laws to simulate what happens when an action is taken.

---

#### 🤖 Agent

The entity that **takes actions** in the environment. In robotics, this is your robot. In code, it's the object that holds your policy (your trained model).

---

#### 🔧 Stable-Baselines3 (SB3)

A **library of ready-made RL algorithms** that already know how to talk to Gymnasium environments. Think of it as a toolbox of pre-built brains (PPO, SAC, etc.) that you can plug straight into any Gymnasium environment.

---

### 1.3 The Full Ecosystem Comparison Table

| Name                  | What It Is                                   | Layer      | Example Usage                        |
| --------------------- | -------------------------------------------- | ---------- | ------------------------------------ |
| **Gym**               | Original OpenAI library (deprecated)         | Interface  | Old code you'll see online           |
| **Gymnasium**         | Fork/successor of Gym (maintained by Farama) | Interface  | Everything new you write             |
| **MuJoCo**            | Physics simulator for rigid body dynamics    | Physics    | Humanoid walking, arm manipulation   |
| **Isaac Lab**         | NVIDIA's GPU-accelerated robot RL framework  | Platform   | Massive parallel robot training      |
| **Stable-Baselines3** | Pre-built RL algorithms (PPO, SAC, TD3)      | Algorithms | Training without building algorithms |

> **Important distinction:** Gymnasium is NOT a physics simulator. MuJoCo IS a physics simulator. Gymnasium wraps MuJoCo environments so your RL algorithm can talk to them through a standard interface.

---

## 2. The Core RL Loop

### 2.1 The Big Picture

Before touching any code, you need to understand the **heartbeat of all RL**. Everything in this guide is a variation of this loop:

```
┌─────────────────────────────────────────────────────────────┐
│                     THE RL LOOP                             │
│                                                             │
│    ┌─────────┐  observation  ┌───────────┐                  │
│    │         │ ◄──────────── │           │                  │
│    │  Agent  │               │Environment│                  │
│    │ (Brain) │ ──────────── ►│           │                  │
│    │         │    action     │           │                  │
│    └─────────┘               └───────────┘                  │
│         ▲                         │                         │
│         │        reward           │                         │
│         └─────────────────────────┘                         │
│                                                             │
│  Repeat this until: done / terminated / truncated           │
└─────────────────────────────────────────────────────────────┘
```

Let's define every single term in this loop precisely.

---

### 2.2 Observation

**What it is:** The information the agent gets about the current state of the world.

**Why it exists:** The agent is blind — it can't directly "see" the environment. It only knows what it's told through observations.

*Balancing robot analogy:* Your robot has sensors. It reads:

- Its current tilt angle (e.g., 3.2 degrees forward)
- Its angular velocity (e.g., rotating at 0.5 rad/s)
- Its joint positions
- Its foot contact forces

All of this sensor data bundled together = the **observation**.

**Key insight:** The agent cannot act on what it doesn't observe. If you don't include foot contact forces in the observation, the agent can't use that information — even if it would help balance.

In code, observations are typically **NumPy arrays** (lists of numbers):

```
observation = [0.032, 0.5, -0.12, 0.0, 0.87, ...]
               ↑       ↑     ↑      ↑    ↑
              tilt   vel  hip_pos  knee  ankle
```

---

### 2.3 Action

**What it is:** The command the agent sends to the environment.

**Why it exists:** The agent needs a way to affect the world. Without actions, there is nothing to learn.

*Walking humanoid analogy:* Actions are the **motor commands** sent to each joint:

- Hip joint: apply 12.3 Nm of torque
- Knee joint: apply -5.1 Nm of torque
- Ankle joint: apply 8.7 Nm of torque

For a humanoid with 17 joints, an action might be a vector of 17 numbers, one torque per joint.

**Two types of action spaces (preview — deep dive in Chapter 3):**

- **Continuous:** A real number (like joint torque). Used in robotics almost always.
- **Discrete:** A choice from a menu (like LEFT, RIGHT, JUMP). Used in games.

---

### 2.4 Reward

**What it is:** A scalar (single) number the environment gives the agent after each action, telling it how good or bad that action was.

**Why it exists:** This is how the agent learns. Over thousands of attempts, the agent figures out that actions leading to high rewards are good, and actions leading to low rewards are bad.

*Walking humanoid analogy:*

- +1.0 reward every timestep the humanoid stays upright and moves forward
- -0.5 penalty if the humanoid falls
- -0.001 per unit of torque used (to encourage energy efficiency)

**Critical insight:** The reward function is arguably the most important design decision in RL. If you reward a robot for moving fast and it discovers it can do that by falling head-first — it will. The reward function encodes what you actually want.

---

### 2.5 Done, Terminated, and Truncated

These three flags tell you when an **episode** has ended. They're easy to confuse but have distinct meanings.

#### Episode

An **episode** is one complete run of your agent in the environment, from a fresh start to some end condition.

*Analogy:* One game of chess from start to checkmate = one episode. One robot walking attempt from standing still until it falls = one episode.

---

#### `terminated` (True/False)

The episode ended because a **natural end condition was reached** — something built into the environment's rules.

*Robotics examples:*

- The humanoid fell over (a failure condition)
- The robot reached its goal position
- The robot's torso touched the ground

This is the environment saying: *"This episode is over because something definitive happened."*

---

#### `truncated` (True/False)

The episode ended because we **ran out of time** — a maximum number of steps was reached — not because anything failed or succeeded.

*Analogy:* A chess game ended not by checkmate but because you ran out of time on the clock.

*Robotics example:* We allowed 1000 steps per episode. The robot made it to step 1000 without falling. Episode ends — not because it failed, but because we artificially cut it off.

**Why truncation exists:** Without a time limit, an agent that never falls never ends its episode. You'd never start a new training episode. Truncation keeps training moving.

---

#### The Old `done` Flag (Gym Legacy)

In the old `gym` library (pre-Gymnasium), there was just one `done` flag that combined `terminated` and `truncated`. **Gymnasium split these** to be more precise. You'll see `done` in old code — mentally translate it as "terminated OR truncated."

```
# Old Gym style (deprecated):
obs, reward, done, info = env.step(action)

# Modern Gymnasium style:
obs, reward, terminated, truncated, info = env.step(action)

# Combining them if you need old-style behavior:
done = terminated or truncated
```

---

### 2.6 Timestep

**What it is:** One single tick of the simulation clock. One step forward in time.

Each timestep:

1. Agent reads observation
2. Agent picks action
3. Environment applies action, simulates physics forward
4. Environment returns new observation + reward + terminated + truncated

*Analogy:* In a movie, a timestep is one frame. The movie plays at 30 frames per second. Similarly, MuJoCo might simulate at 500Hz (500 timesteps per second), but your RL agent might only act at 50Hz (every 10th simulation step). This is called **action repeat** or **frame skipping**.

---

### 2.7 Putting It All Together: CartPole Example

CartPole is the "Hello World" of RL. A pole is balanced on a cart. You can push the cart left or right.

```
       │  ← pole
       │
   ┌───┴───┐
   │ CART  │────► track
   └───────┘
   ← push  push →
```

| Term            | CartPole Meaning                                                  |
| --------------- | ----------------------------------------------------------------- |
| **Observation** | [cart position, cart velocity, pole angle, pole angular velocity] |
| **Action**      | 0 = push left, 1 = push right                                     |
| **Reward**      | +1 for every timestep the pole doesn't fall                       |
| **Terminated**  | Pole angle > 12°, or cart moves off track                         |
| **Truncated**   | 500 steps completed                                               |
| **Episode**     | One balancing attempt, fresh from upright pole                    |

---

## 3. Gymnasium API Deep Dive

### 3.1 Why a Standard API?

Before learning the methods, appreciate *why* they are the way they are.

The Gymnasium API has exactly **5 core methods**. That's it. If you understand these 5 methods, you can work with ANY environment — CartPole, MuJoCo Humanoid, Isaac Lab, custom environments, everything.

---

### 3.2 `gym.make()` — Creating an Environment

**What it does:** Creates and returns an environment object.

**Why it exists:** You need some way to instantiate your environment. `gym.make()` looks up a registered environment by name and builds it.

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
```

Line by line:

- `import gymnasium as gym` — Import the Gymnasium library, alias it as `gym` for convenience
- `gym.make("CartPole-v1")` — Look up the environment registered under the name `"CartPole-v1"` and create it
- `render_mode="human"` — Tell the environment to open a visual window when rendering. Options are `"human"` (window), `"rgb_array"` (returns pixel array), or `None` (no rendering)

**Robotics example:**

```python
env = gym.make("Humanoid-v4", render_mode="human")
env = gym.make("Ant-v4")           # quadruped walking
env = gym.make("HalfCheetah-v4")   # 2D cheetah running
env = gym.make("Hopper-v4")        # 1D hopping robot
```

---

### 3.3 `env.reset()` — Starting a Fresh Episode

**What it does:** Resets the environment to its initial state and returns the first observation.

**Why it exists:** Before each new episode, you need to put the world back to a clean starting state. A fallen robot needs to be stood back up. A game needs to restart.

```python
observation, info = env.reset(seed=42)
```

Line by line:

- `env.reset()` — Wipe the environment state, reset the clock, position the agent at the start
- `seed=42` — Optional. Sets the random seed for reproducibility. With the same seed, the same random events will happen in the same order. Crucial for debugging.
- Returns two things:
  - `observation` — The initial observation (what the agent sees at the very first moment)
  - `info` — A dictionary of optional extra information (can be empty `{}`)

**Important:** Always call `reset()` before the first episode AND after each episode ends.

---

### 3.4 `env.step()` — Taking One Action

**What it does:** Applies an action to the environment, simulates one timestep, and returns the results.

**Why it exists:** This is the core mechanism of interaction. This is how the agent affects the world.

```python
observation, reward, terminated, truncated, info = env.step(action)
```

We'll cover this in extreme detail in Chapter 4. For now, understand that this is called **once per timestep** and drives the entire RL loop.

---

### 3.5 `env.render()` — Visualizing the Environment

**What it does:** Displays the current state of the environment.

**Why it exists:** Humans need to see what's happening for debugging and enjoyment.

```python
env.render()  # Opens window if render_mode="human"
```

- If `render_mode="human"`: Opens/updates a visualization window
- If `render_mode="rgb_array"`: Returns a NumPy array of shape `(height, width, 3)` — a pixel image

**Note:** In modern Gymnasium, rendering happens automatically when you call `step()` if `render_mode="human"`. You don't always need to call `render()` manually.

---

### 3.6 `env.close()` — Cleaning Up

**What it does:** Properly shuts down the environment, freeing memory and closing windows.

**Why it exists:** Physics simulators like MuJoCo run in the background and allocate GPU/CPU resources. Closing properly prevents memory leaks.

```python
env.close()
```

Always call this when you're done. Think of it like `file.close()` when writing to files — you must close what you open.

---

### 3.7 Observation Space and Action Space

Here is one of the most important concepts in Gymnasium.

Every environment has two **spaces** that describe:

1. What observations look like (what shapes/ranges are valid)
2. What actions look like (what shapes/ranges are valid)

These spaces answer questions like:

- How many numbers are in the observation? What are their min/max values?
- Is the action a single integer choice, or a vector of continuous numbers?

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

print(env.observation_space)
# Output: Box([-4.8  -inf  -0.42  -inf], [4.8  inf  0.42  inf], (4,), float32)

print(env.action_space)
# Output: Discrete(2)
```

---

### 3.8 `Box` — Continuous Spaces

**What it is:** A multi-dimensional space of real numbers with defined minimum and maximum values.

**Why it's called Box:** It's literally a box (a bounded region) in N-dimensional space.

```python
from gymnasium.spaces import Box
import numpy as np

# A 1D box: a single number between -1 and 1
space_1d = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

# A 4D box: CartPole's observation space
space_cart = Box(
    low=np.array([-4.8, -np.inf, -0.418, -np.inf]),
    high=np.array([4.8, np.inf, 0.418, np.inf]),
    dtype=np.float32
)

# A 17D box: joint torques for a humanoid (actions)
space_humanoid_action = Box(
    low=-1.0,     # min torque: -1.0 (normalized)
    high=1.0,     # max torque: +1.0 (normalized)
    shape=(17,),  # 17 joints
    dtype=np.float32
)

# Sample a random valid action from this space:
random_action = space_humanoid_action.sample()
# e.g., [ 0.23, -0.87, 0.11, 0.56, -0.33, ... ]  (17 numbers)
```

**Key attributes of Box:**

- `box.low` — Array of minimum values
- `box.high` — Array of maximum values
- `box.shape` — Shape of the array (e.g., `(17,)` or `(64, 64, 3)` for an image)
- `box.sample()` — Returns a random valid sample (useful for random agents)
- `box.contains(x)` — Returns True if `x` is a valid value in this space

---

### 3.9 `Discrete` — Discrete Spaces

**What it is:** A space of a fixed number of integer choices: `{0, 1, 2, ..., n-1}`.

**Why it exists:** Some actions are categorical choices, not continuous numbers.

```python
from gymnasium.spaces import Discrete

# CartPole: 2 actions (push left=0, push right=1)
action_space = Discrete(2)

# A game with 4 directions:
action_space_game = Discrete(4)  # {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT}

# Sample a random action:
random_action = action_space.sample()
# e.g., 1  (just an integer)

print(action_space.n)     # 2 — number of possible actions
print(action_space.sample())  # 0 or 1
```

**Robotics note:** Discrete spaces are rare in robotics. Real robot joints accept continuous torque commands, not discrete choices. Almost all robotics environments use `Box` for actions.

---

### 3.10 `MultiDiscrete` — Multiple Discrete Choices

**What it is:** Multiple independent discrete choices packed together.

```python
from gymnasium.spaces import MultiDiscrete

# Control a robot with 3 separate discrete joints,
# each joint having 5 possible positions:
space = MultiDiscrete([5, 5, 5])  # Each element can be 0,1,2,3,4

sample = space.sample()
# e.g., [3, 1, 4]
```

---

### 3.11 Printing and Exploring Spaces

```python
import gymnasium as gym

env = gym.make("HalfCheetah-v4")

# Explore the observation space
print("=== OBSERVATION SPACE ===")
print(f"Type:  {type(env.observation_space)}")
print(f"Shape: {env.observation_space.shape}")
print(f"Low:   {env.observation_space.low[:5]}...")   # first 5 lows
print(f"High:  {env.observation_space.high[:5]}...")  # first 5 highs
print(f"dtype: {env.observation_space.dtype}")

# Explore the action space
print("\n=== ACTION SPACE ===")
print(f"Type:  {type(env.action_space)}")
print(f"Shape: {env.action_space.shape}")
print(f"Low:   {env.action_space.low}")
print(f"High:  {env.action_space.high}")

# Sample from spaces
obs_sample = env.observation_space.sample()
act_sample = env.action_space.sample()
print(f"\nSample observation shape: {obs_sample.shape}")
print(f"Sample action: {act_sample}")

env.close()
```

**What this prints (approximately for HalfCheetah-v4):**

```
=== OBSERVATION SPACE ===
Type:  <class 'gymnasium.spaces.box.Box'>
Shape: (17,)
Low:   [-inf -inf -inf -inf -inf]...
High:  [inf inf inf inf inf]...
dtype: float64

=== ACTION SPACE ===
Type:  <class 'gymnasium.spaces.box.Box'>
Shape: (6,)
Low:   [-1. -1. -1. -1. -1. -1.]
High:  [1. 1. 1. 1. 1. 1.]
```

*HalfCheetah has 17 observations (joint angles, velocities) and 6 actions (6 joint torques).*

---

## 4. Understanding `env.step()` Deeply

### 4.1 The Most Important Function in Gymnasium

If you truly understand `step()`, you understand 80% of Gymnasium. Let's dissect it completely.

```python
observation, reward, terminated, truncated, info = env.step(action)
```

This single line packs an enormous amount of meaning. Let's break it down from both sides: **what goes in** and **what comes out**.

---

### 4.2 What Goes In: `action`

The action you pass to `step()` must:

1. Be the correct type (NumPy array for Box, integer for Discrete)
2. Be within the valid bounds of `env.action_space`

```python
# For CartPole (Discrete(2)):
action = 0           # push left — just a Python integer
action = 1           # push right

# For Humanoid (Box with shape (17,)):
import numpy as np
action = np.array([0.1, -0.5, 0.3, 0.0, 0.7, ...])  # 17 numbers

# Random valid action (always works):
action = env.action_space.sample()
```

**What happens inside `step()`?**

Internally, the environment:

1. Takes your action
2. Applies it to the physics simulation (adds forces, sets joint targets, etc.)
3. Advances the simulation by one timestep (e.g., 1/500th of a second in MuJoCo)
4. Reads all the sensors
5. Computes the reward
6. Checks termination conditions
7. Returns everything to you

---

### 4.3 What Comes Out: The Five Return Values

#### Return Value 1: `observation`

```python
observation, _, _, _, _ = env.step(action)
```

- **Type:** NumPy array (usually)
- **Shape:** Matches `env.observation_space.shape`
- **Content:** The sensor readings of the environment AFTER the action was applied

*Humanoid example:* After you commanded the hip to apply 12Nm torque, the robot moved. The new observation tells you: where are all the joints now? How fast is the torso moving? Is the foot on the ground?

```python
# For CartPole, observation has 4 numbers:
print(observation)
# e.g., [ 0.0432, -0.205,  0.0361,  0.417]
#         ↑         ↑         ↑         ↑
#      cart_pos  cart_vel  pole_angle  pole_vel
```

---

#### Return Value 2: `reward`

```python
_, reward, _, _, _ = env.step(action)
```

- **Type:** Float (a single number)
- **Range:** Depends entirely on the environment

*CartPole:* `reward = 1.0` for every timestep the pole is upright.

*Humanoid walking:*

```python
reward = (
    + 1.0 * forward_velocity      # go fast forward
    - 0.001 * sum(action**2)      # don't waste energy
    - 0.1 * (torso_height - 1.3)**2  # stay near target height
)
```

**Key insight:** Reward is a scalar — just one number. The entire complexity of "did the robot do well?" gets compressed into a single float. This is both the power and the challenge of RL.

---

#### Return Value 3: `terminated`

```python
_, _, terminated, _, _ = env.step(action)
```

- **Type:** Boolean (True or False)
- **Meaning:** True if a natural terminal condition was reached

| Environment  | `terminated = True` when...                |
| ------------ | ------------------------------------------ |
| CartPole     | Pole angle > 12° OR cart position > 2.4m   |
| Humanoid     | Torso height < 0.8m (fell)                 |
| HalfCheetah  | Never (no failure condition!)              |
| Custom Robot | You define it (e.g., joint limit exceeded) |

**When `terminated = True`:**

- You must call `env.reset()` before continuing
- The episode is truly over — no recovery possible
- The reward for this timestep is still valid and included

---

#### Return Value 4: `truncated`

```python
_, _, _, truncated, _ = env.step(action)
```

- **Type:** Boolean (True or False)
- **Meaning:** True if an artificial time limit was hit

Most environments have a max episode length. CartPole-v1 stops at 500 steps. HalfCheetah-v4 stops at 1000 steps.

**The critical difference between terminated and truncated for learning:**

- `terminated` → The agent caused the episode to end. Clear learning signal: this was a good or bad ending.
- `truncated` → Time ran out. The agent was doing fine but we cut it off. **Not a failure.**

Some RL algorithms treat these differently. For example, when an episode is truncated (not terminated), the value function shouldn't bootstrap to zero — the episode could have continued.

---

#### Return Value 5: `info`

```python
_, _, _, _, info = env.step(action)
```

- **Type:** Dictionary `{}`
- **Content:** Optional extra information that doesn't fit neatly elsewhere

Examples:

```python
# MuJoCo Humanoid info might contain:
info = {
    "x_position": 1.234,           # current x position in world
    "x_velocity": 0.87,            # forward velocity
    "distance_from_origin": 1.234, # how far the robot has walked
    "reward_forward": 0.87,        # breakdown of reward components
    "reward_survive": 1.0,
    "reward_ctrl": -0.023,
}
```

**When to use info:** 

- For debugging (logging detailed metrics)
- For custom reward shaping (your wrapper can read info)
- For evaluation (track distance walked, not just reward)
- NOT for the training loop itself (algorithms shouldn't depend on info)

---

### 4.4 The Complete Step Lifecycle

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

## 5. Writing Your First Gymnasium Program

### 5.1 Setup and Installation

```bash
# Install Gymnasium and dependencies
pip install gymnasium

# For MuJoCo environments (Chapter 6):
pip install gymnasium[mujoco]

# For visualization:
pip install pygame  # needed for CartPole rendering
```

---

### 5.2 Program 1: CartPole with Random Actions

This is the absolute starting point. No learning yet — just a random agent.

```python
# cartpole_random.py

import gymnasium as gym   # Import the Gymnasium library
import time               # Import time, for slowing down the simulation

# ─────────────────────────────────────────────────────────────────
# STEP 1: Create the environment
# ─────────────────────────────────────────────────────────────────
env = gym.make(
    "CartPole-v1",     # Name of the environment to load
    render_mode="human" # "human" = open a window and show animation
)

# ─────────────────────────────────────────────────────────────────
# STEP 2: Run 3 complete episodes
# ─────────────────────────────────────────────────────────────────
NUM_EPISODES = 3

for episode in range(NUM_EPISODES):
    # Reset environment to get the initial observation
    # seed=episode gives different but reproducible randomness each episode
    observation, info = env.reset(seed=episode)

    total_reward = 0.0   # Track total reward for this episode
    step = 0             # Track how many steps we've taken

    print(f"\n{'='*40}")
    print(f"Episode {episode + 1} started")
    print(f"Initial observation: {observation}")
    print(f"{'='*40}")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Step through the environment until the episode ends
    # ─────────────────────────────────────────────────────────────
    while True:
        # Sample a completely random action from the action space
        # For CartPole, this returns 0 or 1 randomly
        action = env.action_space.sample()

        # Apply the action to the environment
        # Get back: new obs, reward, whether episode ended (two ways), extra info
        observation, reward, terminated, truncated, info = env.step(action)

        # Accumulate the total reward for this episode
        total_reward += reward
        step += 1

        # Slow down the simulation so we can actually see it
        time.sleep(0.02)   # 0.02 seconds = 50 FPS, human-visible speed

        # Check if the episode has ended for ANY reason
        if terminated or truncated:
            print(f"Episode {episode + 1} ended!")
            print(f"  Steps taken:   {step}")
            print(f"  Total reward:  {total_reward}")
            print(f"  Reason:        {'Terminated (pole fell)' if terminated else 'Truncated (500 steps)'}")
            break   # Exit the while loop, start next episode

# ─────────────────────────────────────────────────────────────────
# STEP 4: Always close the environment when done
# ─────────────────────────────────────────────────────────────────
env.close()
print("\nDone! Environment closed.")
```

**Expected output:**

```
========================================
Episode 1 started
Initial observation: [ 0.0273956  -0.00611216  0.03585979  0.0197368 ]
========================================
Episode 1 ended!
  Steps taken:   12
  Total reward:  12.0
  Reason:        Terminated (pole fell)

Episode 2 ended!
  Steps taken:   9
  Total reward:  9.0
  Reason:        Terminated (pole fell)

Episode 3 ended!
  Steps taken:   15
  Total reward:  15.0
  Reason:        Terminated (pole fell)
```

*Why is the total reward equal to steps taken? Because CartPole gives exactly +1 per timestep.*

---

### 5.3 Program 2: Pendulum with Continuous Actions

CartPole uses discrete actions (0 or 1). Let's try Pendulum, which uses continuous actions — much closer to real robotics.

**The Pendulum environment:** A pendulum starts hanging down and must be swung up and balanced at the top.

```
Hanging:           Balanced:
   |                  ↑
   ●                  ●
  \↓/                 |
```

The action is a continuous torque applied to the pendulum: any value between -2.0 and +2.0 Nm.

```python
# pendulum_random.py

import gymnasium as gym
import numpy as np    # NumPy for numerical operations
import time

# Create the Pendulum environment
env = gym.make(
    "Pendulum-v1",
    render_mode="human",
    g=9.81   # Gravitational constant (can customize physics!)
)

# Print space info to understand what we're working with
print("=== PENDULUM ENVIRONMENT INFO ===")
print(f"Observation space: {env.observation_space}")
# Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
# Observations: [cos(angle), sin(angle), angular_velocity]

print(f"Action space: {env.action_space}")
# Box(-2.0, 2.0, (1,), float32)
# Action: one continuous torque value between -2.0 and +2.0

NUM_EPISODES = 2

for episode in range(NUM_EPISODES):
    observation, info = env.reset(seed=episode)
    total_reward = 0.0
    step = 0

    print(f"\nEpisode {episode + 1}: Starting...")

    while True:
        # Sample a random continuous action
        # Returns a NumPy array like [1.23] or [-0.87]
        action = env.action_space.sample()

        # Execute the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        # Unpack the observation for readability
        cos_theta, sin_theta, theta_dot = observation
        angle_deg = np.degrees(np.arctan2(sin_theta, cos_theta))  # Recover angle

        if step % 50 == 0:   # Print every 50 steps
            print(f"  Step {step:4d}: action={action[0]:+.3f}, "
                  f"angle={angle_deg:+6.1f}°, "
                  f"reward={reward:.3f}")

        time.sleep(0.01)

        if terminated or truncated:
            print(f"\nEpisode {episode + 1} complete: "
                  f"{step} steps, total_reward={total_reward:.2f}")
            break

env.close()
```

**Key difference from CartPole:**

- Action is `np.array([1.23])` — a 1D NumPy array, not a plain integer
- Reward is negative (between -16.2 and 0): Pendulum rewards being upright and still
- The episode always truncates at 200 steps (no failure condition)

---

### 5.4 Program 3: Multi-Episode Training Loop (Production Pattern)

This is the pattern you'll use in real RL training. Clean, extensible, production-quality.

```python
# training_loop_template.py

import gymnasium as gym
import numpy as np
from collections import deque   # Efficient fixed-size queue for tracking recent rewards

def make_env(env_name: str, seed: int = 0, render: bool = False) -> gym.Env:
    """
    Factory function: creates and returns a configured environment.

    Args:
        env_name: Name of the Gymnasium environment
        seed:     Random seed for reproducibility
        render:   Whether to enable visual rendering

    Returns:
        A configured Gymnasium environment
    """
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    return env


def run_random_agent(
    env_name: str,
    num_episodes: int = 10,
    max_steps_per_episode: int = 500,
    render: bool = False,
    seed: int = 42
) -> dict:
    """
    Runs a random agent for multiple episodes and collects statistics.

    Args:
        env_name:                 Gymnasium environment name
        num_episodes:             How many episodes to run
        max_steps_per_episode:    Safety cap on steps (in case truncation fails)
        render:                   Whether to show visualization
        seed:                     Random seed

    Returns:
        Dictionary of performance statistics
    """
    env = make_env(env_name, seed=seed, render=render)

    # Storage for statistics
    episode_rewards = []       # Total reward per episode
    episode_lengths = []       # Number of steps per episode
    recent_rewards = deque(maxlen=10)  # Last 10 episode rewards for moving average

    print(f"\n{'='*55}")
    print(f"  Running: {env_name}")
    print(f"  Episodes: {num_episodes} | Max steps: {max_steps_per_episode}")
    print(f"  Obs space: {env.observation_space.shape}")
    print(f"  Act space: {env.action_space}")
    print(f"{'='*55}\n")

    for episode in range(num_episodes):
        # Reset: start fresh episode
        observation, info = env.reset(seed=seed + episode)

        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Random agent: sample from action space
            action = env.action_space.sample()

            # Take one step in the environment
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Episode ended naturally or by time limit
            if terminated or truncated:
                break

        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)

        # Print progress
        avg_recent = np.mean(list(recent_rewards))
        print(f"Episode {episode+1:3d}/{num_episodes} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Length: {episode_length:4d} | "
              f"Avg(10): {avg_recent:8.2f}")

    env.close()

    # Compute summary statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward":  np.std(episode_rewards),
        "min_reward":  np.min(episode_rewards),
        "max_reward":  np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"  Mean reward:  {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Min/Max:      {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    print(f"  Mean length:  {stats['mean_length']:.1f} steps")
    print(f"{'='*55}")

    return stats


# ─── Main entry point ────────────────────────────────────────────
if __name__ == "__main__":
    # Test with CartPole first
    stats_cartpole = run_random_agent(
        env_name="CartPole-v1",
        num_episodes=10,
        render=False   # Set to True to see it visually
    )

    # Then Pendulum
    stats_pendulum = run_random_agent(
        env_name="Pendulum-v1",
        num_episodes=5,
        render=False
    )
```

---

## 6. Gymnasium + MuJoCo

### 6.1 What MuJoCo Is

**MuJoCo** stands for **Mu**lti-**Jo**int dynamics with **Co**ntact. It is a physics engine built specifically for simulating robotic and biomechanical systems.

*Analogy:* If Gymnasium is the USB standard (the interface), MuJoCo is the high-performance USB drive (the physics engine). Gymnasium tells you *how* to plug things in; MuJoCo is what actually stores and computes the physics.

**What MuJoCo provides:**

- Rigid body dynamics (bones, links, joints)
- Contact physics (collision detection and response)
- Actuator models (motors, tendons, pneumatics)
- Sensor simulation (accelerometers, gyroscopes, force sensors)
- Extremely fast and accurate simulation

**Why MuJoCo for robotics?**

| Feature     | MuJoCo                  | Why it matters for robots                       |
| ----------- | ----------------------- | ----------------------------------------------- |
| Speed       | Very fast (C++ core)    | Train millions of steps quickly                 |
| Accuracy    | High-fidelity physics   | Policies more likely to transfer to real robots |
| Flexibility | XML-based robot models  | Define any robot joint configuration            |
| Contact     | Stable contact handling | Essential for walking, grasping                 |
| Sensors     | Rich sensor suite       | Realistic observation simulation                |

---

### 6.2 The Architecture: How They Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Your Python Code                                              │
│       │                                                         │
│       ▼                                                         │
│   gymnasium.make("Humanoid-v4")                                 │
│       │                                                         │
│       ▼                                                         │
│   HumanoidEnv (Python class in gymnasium/envs/mujoco/)          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  - Loads humanoid.xml (the robot model)                 │   │
│   │  - Defines observation calculation                      │   │
│   │  - Defines reward function                              │   │
│   │  - Defines termination conditions                       │   │
│   └──────────────┬──────────────────────────────────────────┘   │
│                  │ calls into                                   │
│                  ▼                                              │
│   MuJoCo Physics Engine (C++ library via mujoco-py/mujoco)      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  - Reads the XML model                                  │   │
│   │  - Simulates rigid body dynamics                        │   │
│   │  - Computes joint positions, velocities, forces         │   │
│   │  - Handles contact detection                            │   │
│   │  - Advances simulation time                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6.3 MuJoCo XML Models

Every robot in MuJoCo is described in an **XML file**. This file defines:

- **Bodies:** The physical links (torso, thigh, shin, foot)
- **Joints:** How bodies connect and what motions they allow (hinge, slide, ball)
- **Geoms:** The collision shapes (capsules, boxes, spheres)
- **Actuators:** The motors that can apply forces/torques

Here is a simplified version of a walking leg in MuJoCo XML:

```xml
<!-- simplified_leg.xml — A 3-link walking leg for intuition -->
<mujoco model="simple_leg">

  <!-- Physics settings -->
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <!-- World body — the root, attached to global space -->
  <worldbody>

    <!-- TORSO: The main body -->
    <body name="torso" pos="0 0 1.2">
      <geom type="capsule" size="0.07 0.15" rgba="0.8 0.3 0.3 1"/>
      <!-- Joint that connects torso to world (6 DOF: free floating) -->
      <joint type="free" name="root"/>

      <!-- HIP JOINT BODY: upper leg -->
      <body name="thigh" pos="0 0 -0.15">
        <!-- Hinge joint: rotates around Y axis (swing leg forward/back) -->
        <joint name="hip" type="hinge" axis="0 1 0"
               range="-40 150" damping="0.1"/>
        <geom type="capsule" size="0.05 0.2" rgba="0.3 0.8 0.3 1"/>

        <!-- KNEE JOINT BODY: lower leg -->
        <body name="shin" pos="0 0 -0.4">
          <joint name="knee" type="hinge" axis="0 1 0"
                 range="-140 0" damping="0.1"/>
          <geom type="capsule" size="0.04 0.2" rgba="0.3 0.3 0.8 1"/>

          <!-- FOOT BODY -->
          <body name="foot" pos="0 0 -0.4">
            <joint name="ankle" type="hinge" axis="0 1 0"
                   range="-50 50" damping="0.05"/>
            <geom type="sphere" size="0.06" rgba="0.8 0.8 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- ACTUATORS: Motors that apply torques to joints -->
  <actuator>
    <!-- Each motor controls one joint. gear=100 means apply up to 100Nm -->
    <motor name="hip_motor"   joint="hip"   gear="100"/>
    <motor name="knee_motor"  joint="knee"  gear="80"/>
    <motor name="ankle_motor" joint="ankle" gear="40"/>
  </actuator>

</mujoco>
```

**What each XML element means:**

| XML Element          | What it represents      | Robotics meaning       |
| -------------------- | ----------------------- | ---------------------- |
| `<body>`             | A rigid link            | Bone / structural link |
| `<geom>`             | Collision shape         | Physical surface       |
| `<joint>`            | Degree of freedom       | Knee joint, hip joint  |
| `<actuator>/<motor>` | Force/torque source     | Electric motor         |
| `gear="100"`         | Amplification factor    | Motor strength         |
| `range="-40 150"`    | Joint limits in degrees | Physical stop of joint |
| `damping="0.1"`      | Energy dissipation      | Joint friction         |

---

### 6.4 How Gymnasium Uses MuJoCo Internally

When you call `env.step(action)` on a MuJoCo environment:

```python
# What ACTUALLY happens inside HumanoidEnv.step(action):

def step(self, action):
    # 1. Clip action to valid range (safety)
    action = np.clip(action, self.action_space.low, self.action_space.high)

    # 2. Set the control signals in MuJoCo
    #    self.data.ctrl = actuator control inputs
    self.data.ctrl[:] = action

    # 3. Advance the MuJoCo simulation by n_frames steps
    #    (MuJoCo simulates at 500Hz, RL acts at 50Hz → skip 10 frames)
    for _ in range(self.frame_skip):
        mujoco.mj_step(self.model, self.data)

    # 4. Read sensor data to build observation
    observation = self._get_obs()
    #   → reads joint positions from self.data.qpos
    #   → reads joint velocities from self.data.qvel
    #   → reads accelerometer from self.data.sensordata

    # 5. Compute reward
    x_velocity = (self.data.qpos[0] - x_position_before) / self.dt
    reward = self._compute_reward(action, x_velocity)

    # 6. Check termination
    terminated = self._is_terminated()

    return observation, reward, terminated, False, {}
```

---

### 6.5 Available MuJoCo Environments in Gymnasium

```python
import gymnasium as gym

# Locomotion environments:
env = gym.make("HalfCheetah-v4")  # 2D cheetah-like robot running
env = gym.make("Hopper-v4")       # 1-legged hopping robot
env = gym.make("Walker2d-v4")     # 2D bipedal walker
env = gym.make("Ant-v4")          # 4-legged ant robot (3D)
env = gym.make("Humanoid-v4")     # Full 3D humanoid (most complex)
env = gym.make("HumanoidStandup-v4")  # Humanoid getting up from ground

# Manipulation environments:
env = gym.make("Reacher-v4")      # 2-link arm reaching task
env = gym.make("Pusher-v4")       # Arm pushing object to target
```

**Humanoid-v4 details:**

```python
env = gym.make("Humanoid-v4")
print(env.observation_space.shape)  # (376,) — 376 sensor values!
print(env.action_space.shape)       # (17,)  — 17 joint torques
```

The 376 observations include:

- 22 body part positions
- 23 joint angles
- 23 joint velocities
- 10 contact forces (feet)
- Torso orientation and velocity
- Acceleration and velocity sensors

---

### 6.6 Quick MuJoCo Program

```python
# mujoco_humanoid_random.py

import gymnasium as gym
import numpy as np

# Create the Humanoid environment
env = gym.make("Humanoid-v4", render_mode="human")

print("Humanoid environment loaded!")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space shape:      {env.action_space.shape}")
print(f"Action range:            [{env.action_space.low[0]:.1f}, "
      f"{env.action_space.high[0]:.1f}]")

# Run one episode with random actions
observation, info = env.reset(seed=0)
total_reward = 0.0

for step in range(300):  # 300 steps
    # Random torque commands for all 17 joints
    action = env.action_space.sample()   # shape: (17,)

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 50 == 0:
        print(f"Step {step}: reward={reward:.3f}, "
              f"x_pos={info.get('x_position', 'N/A')}")

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

print(f"Total reward: {total_reward:.2f}")
env.close()
```

---

## 7. Environment Design Internals

### 7.1 What Happens Inside an Environment

When you work with real robotics, you'll need to create or modify environments. To do that well, you need to understand the *internals* of how Gymnasium environments are structured.

Think of an environment as a **machine with 4 subsystems:**

```
┌──────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT MACHINE                       │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  1. Action   │    │  2. Physics  │    │  3. Obs      │    │
│  │  Processing  │───►│  Simulation  │───►│  Generation  │    │
│  └──────────────┘    └──────────────┘    └──────┬───────┘    │
│                                                 │            │
│                        ┌──────────────┐         │            │
│                        │  4. Reward + │◄────────┘            │
│                        │  Termination │                      │
│                        └──────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

Let's understand each subsystem.

---

### 7.2 Subsystem 1: Action Processing

**What it does:** Takes the raw action from the agent and translates it into forces/commands for the physics simulation.

**Why processing is needed:**

- RL algorithms output numbers in normalized ranges (e.g., [-1, 1])
- Physical actuators need real units (Nm of torque, N of force)
- You may want to clip, scale, or add noise

```python
def _process_action(self, raw_action: np.ndarray) -> np.ndarray:
    """
    Example action processing for a robot arm.

    Args:
        raw_action: Output from RL agent, typically in [-1, 1]

    Returns:
        Actual joint torques in Nm
    """
    # Step 1: Clip to valid range (safety)
    clipped = np.clip(raw_action, -1.0, 1.0)

    # Step 2: Scale from [-1,1] to actual torque range
    # e.g., max torque = 50Nm, so multiply by 50
    max_torque = np.array([50.0, 80.0, 30.0, 80.0, 50.0, 20.0])
    torques = clipped * max_torque

    # Step 3: Optionally add Gaussian noise (simulate motor noise)
    # noise = np.random.normal(0, 0.01, size=torques.shape)
    # torques += noise

    return torques
```

---

### 7.3 Subsystem 2: Physics Simulation (The Engine)

For custom environments, you can implement physics yourself (simple math) or use a simulator (MuJoCo). For the custom example in Chapter 8, we'll use simple math.

For MuJoCo-based environments, this step is handled by `mujoco.mj_step()`.

---

### 7.4 Subsystem 3: Observation Generation

**What it does:** Reads the current state of the simulation and packages it as a NumPy array for the agent.

**Key design decisions:**

- What does the agent NEED to know to do its job?
- What is MEASURABLE on a real robot? (for sim-to-real transfer)
- What units should observations be in?
- Should you normalize observations?

```python
def _get_obs(self) -> np.ndarray:
    """
    Build the observation vector from simulation state.

    Robot state variables (from MuJoCo data):
    - self.data.qpos: joint positions (angles in radians)
    - self.data.qvel: joint velocities (rad/s)
    - self.data.sensordata: sensor readings (forces, accelerometers)
    """
    # Joint positions (angles), skipping the first 7 (root position/orientation)
    joint_angles = self.data.qpos[7:].copy()          # shape: (n_joints,)

    # Joint velocities
    joint_velocities = self.data.qvel[6:].copy()       # shape: (n_joints,)

    # Torso orientation as quaternion [w, x, y, z]
    torso_orientation = self.data.qpos[3:7].copy()     # shape: (4,)

    # Torso linear velocity [vx, vy, vz]
    torso_velocity = self.data.qvel[:3].copy()         # shape: (3,)

    # Foot contact forces (are feet touching the ground?)
    contact_forces = self.data.sensordata[:4].copy()   # shape: (4,)

    # Concatenate all into one flat vector
    observation = np.concatenate([
        joint_angles,        # Where are the joints?
        joint_velocities,    # How fast are they moving?
        torso_orientation,   # How is the body tilted?
        torso_velocity,      # How fast is the body moving?
        contact_forces,      # Are feet on the ground?
    ])

    return observation.astype(np.float64)
```

**Observation design principles for robotics:**

1. **Include velocity:** Position alone is usually insufficient. The agent needs to know how fast things are moving to predict future states.
2. **Use sensor readings that exist on real robots:** Don't include "ground truth" position from the simulator if the real robot won't have GPS.
3. **Normalize:** Keep values roughly in [-1, 1] for training stability. RL neural networks hate inputs with wildly different scales.

---

### 7.5 Subsystem 4: Reward Function Design

The reward function is the **most consequential design decision** in your environment. Get it wrong, and your robot will find bizarre solutions to maximize reward without doing what you actually want.

**The Reward Hacking Problem:**

> You reward a cleaning robot +1 for each piece of dirt it touches.
> The robot learns to pick up dirt, drop it, and pick it up again. Infinite reward!

**Principles for good reward design:**

#### Principle 1: Reward what you actually want

```python
# BAD: Rewards distance moved (robot might roll forward)
reward = current_x_position - previous_x_position

# BETTER: Rewards walking speed, punishes falling
reward = (
    + 1.0 * forward_velocity      # encourage moving forward
    - 100.0 * float(has_fallen)   # strongly discourage falling
)
```

#### Principle 2: Shaping rewards — dense vs sparse

**Sparse reward:** Only +1 at the end, 0 everywhere else.

```python
# Sparse: only reward when goal is reached
reward = 1.0 if distance_to_goal < 0.1 else 0.0
```

*Problem:* Very hard to learn. The agent almost never gets reward signal.

**Dense reward:** Reward every step based on progress.

```python
# Dense: reward proportional to progress toward goal
reward = -distance_to_goal  # closer = less negative = better
```

*Better for learning, but must be designed carefully.*

#### Principle 3: Multiple reward components

```python
def compute_reward(self, action, info) -> float:
    """
    Multi-component reward for humanoid walking.
    """
    # Component 1: Forward progress (main objective)
    r_forward = 1.0 * info["x_velocity"]

    # Component 2: Survival bonus (stay alive)
    r_survive = 5.0

    # Component 3: Energy efficiency (don't waste power)
    r_ctrl = -0.1 * np.sum(action ** 2)

    # Component 4: Upright posture (don't lean too much)
    r_upright = -2.0 * abs(info["torso_tilt_angle"])

    # Component 5: Contact symmetry (step left and right equally)
    r_symmetry = -0.5 * abs(info["left_foot_contact"] - info["right_foot_contact"])

    total_reward = r_forward + r_survive + r_ctrl + r_upright + r_symmetry

    return total_reward
```

---

### 7.6 Termination Conditions

```python
def _is_terminated(self) -> bool:
    """
    Check if the episode should end due to a failure or success.
    """
    # Read current state
    torso_height = self.data.qpos[2]      # z position of torso
    torso_tilt = self._get_torso_tilt()   # how much is it leaning?

    # Failure conditions
    fell_over = torso_height < 0.8         # torso too low = fallen
    too_tilted = abs(torso_tilt) > 0.7    # too tilted = falling
    joint_limit = self._joint_at_limit()   # hit physical stop

    return bool(fell_over or too_tilted or joint_limit)
```

---

## 8. Building a Custom Environment

### 8.1 The Goal

We're building: **a simple 1D balancing robot.**

```
    ↑ pole angle (θ)
     \
      \
   ┌───●───┐   ← robot body
   │       │
   [=======]    ← ground (fixed base)

The pole must stay upright. The agent can apply torque to the pole joint.
```

This is a simplified version of what a real humanoid torso balancing problem looks like.

---

### 8.2 The Custom Environment Template

Every custom Gymnasium environment must:

1. Inherit from `gymnasium.Env`
2. Define `observation_space` and `action_space`
3. Implement `reset(self, seed=None, options=None)`
4. Implement `step(self, action)`
5. Optionally implement `render()` and `close()`

```python
# simple_balancer.py
# A simple pole-balancing robot — your first custom Gymnasium environment

import gymnasium as gym           # The Gymnasium base class and registration tools
from gymnasium import spaces      # For defining observation/action spaces
import numpy as np                # For numerical arrays and math
from typing import Optional, Tuple  # Type hints for clarity


class SimpleBalancerEnv(gym.Env):
    """
    A simple 1D pole-balancing environment.

    The Problem:
        A pole is attached to a fixed base via a hinge joint.
        The pole starts nearly upright and will fall due to gravity.
        The agent must apply torques to keep the pole balanced upright.

    Observation (4 values):
        - pole_angle       (radians): Current angle from vertical. 0 = upright.
        - pole_velocity    (rad/s):   How fast the pole is rotating.
        - pole_sin         (unitless): sin(angle), gives direction info
        - pole_cos         (unitless): cos(angle), gives tilt info

    Action (1 value, continuous):
        - torque [-2, +2] Nm: Torque to apply to the pole joint.
          Positive = push pole to the right, Negative = push left.

    Reward:
        - +1.0 per timestep the pole stays upright (|angle| < 0.2 rad)
        - -0.01 per unit of torque used (encourage energy efficiency)
        - -1.0 bonus penalty if the pole falls

    Termination:
        - Pole angle exceeds ±1.0 radian (≈57°) — it has fallen

    Truncation:
        - After 500 timesteps
    """

    # ─────────────────────────────────────────────────────────────
    # IMPORTANT: Tell Gymnasium what render modes we support
    # ─────────────────────────────────────────────────────────────
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Constructor: called once when you do gym.make() or SimpleBalancerEnv().

        Set up all the constants, physics parameters, and spaces here.
        Do NOT run any simulation here — only configure.

        Args:
            render_mode: How to visualize. "human" = print to terminal.
        """
        super().__init__()   # Always call parent __init__

        # ── Physics constants ──────────────────────────────────────
        self.gravity     = 9.81   # m/s²
        self.pole_length = 1.0    # meters, from pivot to tip
        self.pole_mass   = 0.5    # kg, pole mass
        self.max_torque  = 2.0    # Nm, maximum applicable torque
        self.dt          = 0.02   # seconds per timestep (50 Hz control)

        # Damping: energy dissipation (simulates friction in the joint)
        self.damping = 0.05

        # Episode parameters
        self.max_episode_steps = 500

        # ── Define the OBSERVATION SPACE ──────────────────────────
        # The agent sees 4 continuous values
        # We define their bounds (min, max for each)
        obs_low  = np.array([-np.pi, -10.0, -1.0, -1.0], dtype=np.float32)
        obs_high = np.array([ np.pi,  10.0,  1.0,  1.0], dtype=np.float32)
        #                      ↑         ↑       ↑      ↑
        #                  angle    velocity  sin    cos

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # ── Define the ACTION SPACE ───────────────────────────────
        # The agent can apply a continuous torque in [-max_torque, +max_torque]
        self.action_space = spaces.Box(
            low=np.array([-self.max_torque], dtype=np.float32),
            high=np.array([self.max_torque], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        # ── Internal state variables ──────────────────────────────
        # These will be set in reset(). Initialize to None here.
        self.angle    = None   # Current pole angle (radians)
        self.velocity = None   # Current pole angular velocity (rad/s)
        self.step_count = None # Track how many steps taken in this episode

        # Store render mode
        self.render_mode = render_mode

    # ─────────────────────────────────────────────────────────────
    # reset(): Called at the START of every episode
    # ─────────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to a fresh initial state.

        Always call super().reset(seed=seed) first — this sets up
        the random number generator (self.np_random) for reproducibility.

        Args:
            seed:    Random seed. Same seed → same initial conditions.
            options: Optional extra configuration (can ignore for now).

        Returns:
            observation: The initial observation (what agent sees at step 0)
            info:        Empty dict (or optional extra info)
        """
        # REQUIRED: Initialize the RNG. Allows seeded reproducibility.
        super().reset(seed=seed)

        # Set initial pole angle: small random perturbation near upright (0 rad)
        # np_random is set up by super().reset(seed=seed)
        self.angle = self.np_random.uniform(
            low=-0.1,    # -0.1 radians ≈ -5.7 degrees
            high=0.1     # +0.1 radians ≈ +5.7 degrees
        )

        # Set initial velocity: small random perturbation near zero
        self.velocity = self.np_random.uniform(low=-0.1, high=0.1)

        # Reset step counter
        self.step_count = 0

        # Build and return the initial observation
        observation = self._get_obs()
        info = {}   # Empty info dict for now

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ─────────────────────────────────────────────────────────────
    # step(): Called every timestep
    # ─────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply one action, advance physics, compute reward, check termination.

        Args:
            action: NumPy array of shape (1,) with torque in [-2, +2]

        Returns:
            observation: New state after action
            reward:      Scalar reward signal
            terminated:  True if pole has fallen (natural failure)
            truncated:   True if max steps reached (time limit)
            info:        Extra information dictionary
        """
        # ── Step 1: Process the action ────────────────────────────
        # Extract the torque value from the action array
        torque = float(action[0])

        # Clip torque to valid range (safety guard)
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # ── Step 2: Physics simulation (Euler integration) ────────
        # We're using the equation of motion for an inverted pendulum:
        # α = (τ - m*g*L*sin(θ) - d*ω) / (m*L²)
        # Where:
        #   α = angular acceleration (what we compute)
        #   τ = applied torque
        #   m = pole mass
        #   g = gravity
        #   L = pole length
        #   θ = current angle
        #   ω = current angular velocity
        #   d = damping coefficient

        m = self.pole_mass
        g = self.gravity
        L = self.pole_length
        d = self.damping

        # Gravity pulls the pole toward vertical (restoring force)
        gravity_torque = m * g * L * np.sin(self.angle)

        # Damping opposes motion (friction)
        damping_force = d * self.velocity

        # Net angular acceleration
        angular_acceleration = (torque - gravity_torque - damping_force) / (m * L**2)

        # Euler integration: update velocity and angle
        # New velocity = old velocity + acceleration × time_step
        self.velocity += angular_acceleration * self.dt

        # New angle = old angle + velocity × time_step
        self.angle += self.velocity * self.dt

        # Keep angle in [-π, π] range (wrap around)
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))

        # Advance the step counter
        self.step_count += 1

        # ── Step 3: Compute reward ────────────────────────────────
        # Upright bonus: reward for staying near vertical
        # cos(angle) is 1.0 when perfectly upright, -1.0 when inverted
        # This naturally rewards the upright position
        upright_reward = np.cos(self.angle)   # in [-1, +1]

        # Energy penalty: penalize large control actions
        # Encourages the agent to find efficient solutions
        energy_penalty = -0.001 * (torque ** 2)

        # Velocity penalty: penalize large, jerky movements
        velocity_penalty = -0.001 * (self.velocity ** 2)

        # Total reward
        reward = float(upright_reward + energy_penalty + velocity_penalty)

        # ── Step 4: Check termination ─────────────────────────────
        # The pole has "fallen" if it tips more than ~57 degrees
        angle_threshold = 1.0  # radians ≈ 57 degrees

        fallen = bool(abs(self.angle) > angle_threshold)
        terminated = fallen   # Episode ends if pole falls

        # ── Step 5: Check truncation ──────────────────────────────
        truncated = bool(self.step_count >= self.max_episode_steps)

        # ── Step 6: Build observation ─────────────────────────────
        observation = self._get_obs()

        # ── Step 7: Build info dict ───────────────────────────────
        # Include useful debugging/logging information
        info = {
            "angle_deg":   np.degrees(self.angle),  # angle in degrees (human-readable)
            "velocity":    self.velocity,
            "torque":      torque,
            "step_count":  self.step_count,
            "upright_reward": upright_reward,
            "energy_penalty": energy_penalty,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────
    # _get_obs(): Helper to build observation vector
    # ─────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """
        Construct the observation array from current state.

        We include sin and cos of the angle rather than the raw angle
        because sin/cos don't have a discontinuity at ±π, making learning easier.

        Returns:
            Observation array of shape (4,) with dtype float32
        """
        return np.array([
            self.angle,              # Current pole angle
            self.velocity,           # Current angular velocity
            np.sin(self.angle),      # Sine of angle (direction info)
            np.cos(self.angle),      # Cosine of angle (tilt magnitude)
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────
    # render(): Visualize the current state
    # ─────────────────────────────────────────────────────────────
    def render(self):
        """
        Simple text-based visualization.
        For production, you'd use Pygame or Matplotlib here.
        """
        if self.render_mode == "ansi":
            angle_deg = np.degrees(self.angle)
            bar_pos = int((self.angle / np.pi) * 20) + 20  # Map to 40-char bar
            bar_pos = np.clip(bar_pos, 0, 39)
            bar = [' '] * 40
            bar[20] = '|'   # Center = upright
            bar[bar_pos] = '●'  # Current pole tip position
            bar_str = ''.join(bar)
            print(f"[{bar_str}] angle={angle_deg:+6.1f}° vel={self.velocity:+5.2f}")

        elif self.render_mode == "human":
            # For a simple human-readable display
            angle_deg = np.degrees(self.angle)
            status = "UPRIGHT" if abs(self.angle) < 0.2 else "TILTING"
            if abs(self.angle) > 0.8:
                status = "FALLING!"
            print(f"Step {self.step_count:4d} | angle={angle_deg:+6.1f}° | "
                  f"vel={self.velocity:+5.2f} | {status}")

    def close(self):
        """Clean up resources. Called when done with the environment."""
        pass   # Nothing to clean up in this simple environment
```

---

### 8.3 Registering and Testing the Custom Environment

```python
# test_custom_env.py

import gymnasium as gym
import numpy as np
from simple_balancer import SimpleBalancerEnv  # Import our custom env

# ─────────────────────────────────────────────────────────────────
# Option A: Use directly (without registration)
# ─────────────────────────────────────────────────────────────────
env = SimpleBalancerEnv(render_mode="ansi")  # "ansi" = text visualization

# Verify spaces are correct
print("=== CUSTOM ENV VERIFICATION ===")
print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")

# Verify space contents are valid
sample_obs = env.observation_space.sample()
sample_act = env.action_space.sample()
print(f"Sample obs: {sample_obs}")
print(f"Sample act: {sample_act}")

# Check that observation space contains valid observations
obs, info = env.reset(seed=42)
print(f"\nInitial obs: {obs}")
print(f"In obs space? {env.observation_space.contains(obs)}")  # Should be True

# ─────────────────────────────────────────────────────────────────
# Run one episode with random actions
# ─────────────────────────────────────────────────────────────────
print("\n=== RUNNING RANDOM EPISODE ===")

obs, info = env.reset(seed=0)
total_reward = 0.0
step = 0

while True:
    action = env.action_space.sample()   # Random torque
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1

    # Print every 20 steps
    if step % 20 == 0:
        print(f"  Step {step:3d}: reward={reward:.3f}, "
              f"angle={info['angle_deg']:+.1f}°")

    if terminated or truncated:
        reason = "Pole fell!" if terminated else "Time limit"
        print(f"\n  Episode ended: {reason}")
        print(f"  Steps: {step}, Total reward: {total_reward:.2f}")
        break

env.close()

# ─────────────────────────────────────────────────────────────────
# Option B: Register and use with gym.make()
# ─────────────────────────────────────────────────────────────────
gym.register(
    id="SimpleBalancer-v0",           # Unique name for this environment
    entry_point=SimpleBalancerEnv,    # The class to instantiate
    max_episode_steps=500,            # Truncation limit (adds TimeLimit wrapper)
)

# Now use it just like any built-in environment!
env2 = gym.make("SimpleBalancer-v0", render_mode=None)
obs, info = env2.reset(seed=7)
print(f"\nRegistered env initial obs: {obs}")
env2.close()
```

---

### 8.4 Validating Your Environment

Gymnasium includes a utility to check your environment for common mistakes:

```python
from gymnasium.utils.env_checker import check_env
from simple_balancer import SimpleBalancerEnv

env = SimpleBalancerEnv()

# This function checks:
# - observation_space is defined correctly
# - action_space is defined correctly
# - reset() returns valid observation
# - step() returns 5 values in correct format
# - observations are within observation_space bounds
# - rewards are scalars
# Prints warnings/errors if anything is wrong
check_env(env, warn=True)

print("Environment check passed! ✓")
env.close()
```

---

## 9. Gymnasium + PPO (Stable-Baselines3)

### 9.1 Why You Need Algorithms

So far, we've only used random agents. Random agents don't learn. To actually train a robot to do something useful, you need an RL algorithm.

**Writing RL algorithms from scratch is hard.** PPO, SAC, and TD3 are hundreds of lines of carefully tuned code involving:

- Policy networks (neural networks mapping observations to actions)
- Value networks (estimating how good a state is)
- Gradient clipping
- Advantage estimation
- Experience replay buffers
- Entropy regularization

**Stable-Baselines3 (SB3)** provides all of these, pre-built, tested, and Gymnasium-compatible.

---

### 9.2 Installation

```bash
pip install stable-baselines3[extra]
# [extra] includes: tensorboard, hugging-face, etc.
```

---

### 9.3 PPO: Proximal Policy Optimization

PPO is the most commonly used RL algorithm for continuous control (robotics). It's reliable, sample-efficient, and parallelizable.

**Intuition for PPO:**

The agent has a **policy**: a function that maps observations to actions.
PPO updates this policy gradually — not too much at once — to prevent catastrophic forgetting.

*Analogy:* Training a dog. You don't change everything at once. You make small corrections: "yes, that step was good," "no, that step wasn't." PPO does the same mathematically, bounding policy updates to be "not too different from what we had before."

---

### 9.4 Training CartPole with PPO

```python
# train_cartpole_ppo.py

import gymnasium as gym                          # The environment interface
from stable_baselines3 import PPO               # The PPO algorithm
from stable_baselines3.common.evaluation import evaluate_policy  # Evaluation utility

# ─────────────────────────────────────────────────────────────────
# STEP 1: Create the training environment
# ─────────────────────────────────────────────────────────────────
# No render_mode during training — rendering slows down training dramatically
train_env = gym.make("CartPole-v1")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Create the PPO model
# ─────────────────────────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",    # "MlpPolicy" = Multi-Layer Perceptron (standard neural network)
                           # Other options: "CnnPolicy" (for images), "MultiInputPolicy"
    env=train_env,         # The environment to train on
    verbose=1,             # 1 = print training progress, 0 = silent
    learning_rate=3e-4,    # How fast the neural network weights change
                           # Too high: unstable. Too low: slow. 3e-4 is a good default.
    n_steps=2048,          # How many steps to collect before each policy update
                           # Larger = more stable gradient estimates but slower updates
    batch_size=64,         # Mini-batch size for gradient updates
    n_epochs=10,           # How many times to reuse each batch of experience
    gamma=0.99,            # Discount factor: how much to value future rewards
                           # 0.99 means reward 100 steps from now is worth 0.99^100 ≈ 0.37 today
    seed=42,               # Reproducibility
)

# ─────────────────────────────────────────────────────────────────
# STEP 3: Train the model
# ─────────────────────────────────────────────────────────────────
print("Starting training...")

model.learn(
    total_timesteps=50_000,    # Train for 50,000 environment steps
    progress_bar=True          # Show progress bar (requires tqdm: pip install tqdm)
)

print("Training complete!")

# ─────────────────────────────────────────────────────────────────
# STEP 4: Save the trained model
# ─────────────────────────────────────────────────────────────────
model.save("cartpole_ppo_model")
print("Model saved to: cartpole_ppo_model.zip")

# ─────────────────────────────────────────────────────────────────
# STEP 5: Evaluate the trained model
# ─────────────────────────────────────────────────────────────────
eval_env = gym.make("CartPole-v1")

# Run 10 evaluation episodes and compute mean/std reward
mean_reward, std_reward = evaluate_policy(
    model,          # The trained policy
    eval_env,       # Environment to evaluate on
    n_eval_episodes=10,   # Number of evaluation episodes
    deterministic=True    # Use deterministic actions (best action, not stochastic)
)

print(f"\nEvaluation results:")
print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
# Perfect CartPole = 500 (max episode length). Well-trained PPO should hit ~500.

eval_env.close()
train_env.close()
```

---

### 9.5 Watching the Trained Agent

```python
# watch_trained_agent.py

import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load the previously saved model
model = PPO.load("cartpole_ppo_model")

# Create environment with visual rendering
env = gym.make("CartPole-v1", render_mode="human")

# Run 5 episodes with the trained policy
for episode in range(5):
    obs, info = env.reset(seed=episode)
    total_reward = 0.0
    step = 0

    while True:
        # Use the trained model to predict the best action
        action, _states = model.predict(
            obs,              # Current observation
            deterministic=True  # Always pick the best action (not explore)
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        time.sleep(0.02)  # Slow down for visibility

        if terminated or truncated:
            print(f"Episode {episode+1}: {step} steps, reward={total_reward:.1f}")
            break

env.close()
```

---

### 9.6 Training Humanoid Walking with PPO

Now apply this to the actual MuJoCo Humanoid:

```python
# train_humanoid_ppo.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import make_vec_env   # Vectorized environments
from stable_baselines3.common.callbacks import EvalCallback  # Auto-evaluation during training
import os

# ─────────────────────────────────────────────────────────────────
# VECTORIZED ENVIRONMENTS — Training on multiple envs in parallel
# ─────────────────────────────────────────────────────────────────
# Running N environments simultaneously makes training N times faster.
# Gymnasium episodes reset at different times, giving more diverse experience.
# This is critical for sample efficiency with complex robots.

N_ENVS = 8   # Run 8 environments simultaneously

# make_vec_env creates N parallel copies of the environment
# SubprocVecEnv: each env runs in its own subprocess (uses multiple CPU cores)
vec_env = make_vec_env(
    "Humanoid-v4",
    n_envs=N_ENVS,
    seed=0
)

print(f"Vectorized env created: {N_ENVS} parallel Humanoid environments")
print(f"Obs space: {vec_env.observation_space.shape}")   # Same as single env
print(f"Act space: {vec_env.action_space.shape}")

# ─────────────────────────────────────────────────────────────────
# Create PPO model for continuous humanoid control
# ─────────────────────────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,

    # Network architecture: two hidden layers of 256 units each
    # Good starting point for locomotion tasks
    policy_kwargs=dict(net_arch=[256, 256]),

    # Larger n_steps for complex tasks (humanoid needs more rollout for stable gradients)
    n_steps=2048,
    batch_size=256,
    n_epochs=10,

    learning_rate=3e-4,
    gamma=0.99,           # High discount (humanoid rewards span many steps)
    gae_lambda=0.95,      # GAE lambda: controls bias-variance tradeoff in advantage estimation
    clip_range=0.2,       # PPO clipping parameter: limits policy update size
    ent_coef=0.0,         # Entropy coefficient: encourages exploration (0 = none)

    tensorboard_log="./humanoid_tensorboard/",  # For live training plots
    seed=42,
)

# ─────────────────────────────────────────────────────────────────
# Set up evaluation callback (evaluates every 50k steps)
# ─────────────────────────────────────────────────────────────────
eval_env = make_vec_env("Humanoid-v4", n_envs=1, seed=999)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_humanoid_model/",  # Save best model automatically
    log_path="./humanoid_eval_logs/",
    eval_freq=50_000,         # Evaluate every 50,000 steps
    n_eval_episodes=5,        # 5 evaluation episodes per evaluation
    deterministic=True,
    verbose=1
)

# ─────────────────────────────────────────────────────────────────
# Train for 5 million steps (humanoid needs a LOT of steps)
# ─────────────────────────────────────────────────────────────────
print("\nStarting Humanoid training...")
print("Note: Full humanoid training typically needs 10-50M steps.")
print("This will take hours on CPU, minutes on GPU.")

model.learn(
    total_timesteps=5_000_000,
    callback=eval_callback,
    progress_bar=True
)

model.save("humanoid_ppo_final")
print("Humanoid training complete! Model saved.")

vec_env.close()
eval_env.close()
```

---

### 9.7 Why Vectorized Environments?

This is a crucial concept for serious robotics RL training:

```
Without vectorized envs (1 environment):
  Collecting 2048 steps = 2048 sequential steps
  Time: 2048 × 0.002s = ~4 seconds per update

With 8 vectorized envs:
  Collecting 2048 steps across 8 envs = 2048 × 8 = 16,384 total steps
  Time: ~4 seconds (parallel!)
  Effective speedup: 8×

Isaac Lab (GPU parallel):
  Can run 4096+ environments simultaneously on a single GPU!
  Training that takes weeks on CPU → hours on GPU
```

---

## 10. The Robotics Perspective: From Sim to Real

### 10.1 How Large Robotics Labs Structure RL Systems

Understanding the full pipeline that companies like Boston Dynamics, Figure, 1X, Agility Robotics, and academia use:

```
┌────────────────────────────────────────────────────────────────────┐
│              FULL ROBOTICS DRL PIPELINE                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  1. ROBOT MODEL                                              │  │
│  │    ● URDF/MJCF XML: joints, links, actuators, sensors        │  │
│  │    ● Inertia parameters, mass distribution                   │  │
│  │    ● Actuator limits, gear ratios                            │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  2. SIMULATION ENVIRONMENT                                   │  │
│  │    ● MuJoCo / Isaac Lab / Genesis                            │  │
│  │    ● 4096+ parallel environments (GPU)                       │  │
│  │    ● Domain randomization (physics variations)               │  │
│  │    ● Gymnasium-compatible interface                          │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  3. REWARD FUNCTION & CURRICULUM                             │  │
│  │    ● Multi-component rewards                                 │  │
│  │    ● Curriculum learning (easy → hard)                       │  │
│  │    ● Constraints (safety bounds, energy limits)              │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  4. RL ALGORITHM                                             │  │
│  │    ● PPO / SAC / TD3 / ARS / TRPO                            │  │
│  │    ● Trained on GPU cluster (millions of steps/hour)         │  │
│  │    ● TensorBoard monitoring                                  │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  5. POLICY (Trained Neural Network)                          │  │
│  │    ● Input: sensor observations                              │  │
│  │    ● Output: joint commands (torques or target angles)       │  │
│  │    ● Typical: 2-3 hidden layers, 256-512 units               │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  6. SIM-TO-REAL TRANSFER                                     │  │
│  │    ● Domain randomization (masses, friction, latency)        │  │
│  │    ● Actuator modeling (motor dynamics, backlash)            │  │
│  │    ● Observation noise injection                             │  │
│  │    ● Action delay simulation                                 │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────────────────┐  │
│  │  7. REAL ROBOT DEPLOYMENT                                    │  │
│  │    ● Policy runs at 50-200 Hz on onboard computer            │  │
│  │    ● Low-level controllers bridge policy → motors            │  │
│  │    ● Safety monitors override policy if needed               │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

### 10.2 Isaac Lab: The GPU-Accelerated Version

Isaac Lab is NVIDIA's robotics learning platform. It's Gymnasium-compatible but massively more powerful:

| Feature        | Gymnasium + MuJoCo | Isaac Lab           |
| -------------- | ------------------ | ------------------- |
| Parallel envs  | ~16 (CPU)          | 4096+ (GPU)         |
| Steps/second   | ~10,000            | ~1,000,000+         |
| Hardware       | CPU                | NVIDIA GPU          |
| Focus          | Research           | Production robotics |
| Robot models   | Preset             | Any URDF            |
| SB3 compatible | Yes                | Yes (wrapper)       |

```python
# Isaac Lab example (conceptual — requires NVIDIA GPU + Isaac Lab installation)
# The API is deliberately Gymnasium-compatible

from isaaclab.envs import ManagerBasedRLEnvCfg
import gymnasium as gym

# Isaac Lab environments are Gymnasium-compatible
# So all our Gymnasium code works directly!
env = gym.make("Isaac-Velocity-Rough-Anymal-D-v0")

# The API is IDENTICAL to what you've learned:
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

---

### 10.3 Domain Randomization for Sim-to-Real

The biggest challenge in robotics RL is the **sim-to-real gap**: policies trained in simulation often fail on real robots because the simulation is not perfectly accurate.

**Domain randomization** is the solution: randomize physics parameters during training so the policy learns to be robust to variations.

```python
# domain_randomization_example.py

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation, TimeLimit


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Wraps an environment and randomizes physics parameters at each reset.

    This teaches the policy to be robust to parameter uncertainty,
    which helps it transfer from simulation to real hardware.

    Randomized parameters:
    - Mass: real robot mass is never exactly what the XML says
    - Friction: floor friction varies with surface
    - Action delay: real motors have control latency
    - Observation noise: real sensors have noise
    """

    def __init__(self, env: gym.Env, randomize: bool = True):
        """
        Args:
            env:        The base Gymnasium environment to wrap
            randomize:  If False, use nominal parameters (for evaluation)
        """
        super().__init__(env)
        self.randomize = randomize

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.randomize:
            self._apply_randomization()

        return obs, info

    def _apply_randomization(self):
        """
        Randomize physics parameters at the start of each episode.

        For MuJoCo environments, we can directly modify model parameters.
        These modifications persist until the next reset.
        """
        if hasattr(self.env, 'model'):
            # Randomize body masses ±20% from nominal
            # This simulates payload uncertainty, worn parts, etc.
            nominal_masses = self.env.model.body_mass.copy()
            self.env.model.body_mass[:] = nominal_masses * np.random.uniform(
                0.8, 1.2, size=nominal_masses.shape
            )

            # Randomize friction coefficient ±30%
            nominal_friction = self.env.model.geom_friction.copy()
            self.env.model.geom_friction[:] = nominal_friction * np.random.uniform(
                0.7, 1.3, size=nominal_friction.shape
            )

    def step(self, action):
        # Add observation noise to simulate real sensors
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.randomize:
            # Add small Gaussian noise to observations
            # Real sensors have measurement noise
            noise = np.random.normal(0, 0.01, size=obs.shape).astype(np.float32)
            obs = obs + noise

            # Clip to valid observation space bounds
            obs = np.clip(obs, self.env.observation_space.low,
                               self.env.observation_space.high)

        return obs, reward, terminated, truncated, info


# Usage:
base_env = gym.make("Hopper-v4")
randomized_env = DomainRandomizationWrapper(base_env, randomize=True)

# Now train on randomized_env — policy will be more robust!
obs, info = randomized_env.reset(seed=42)
print(f"Randomized obs: {obs[:4]}")
randomized_env.close()
```

---

### 10.4 The Observation Design for Real Robots

When designing observations for robots that will eventually be deployed on hardware, only include observations that are **measurable on the real robot**:

```python
# good_observation_design.py — What real humanoids can actually measure

class RealisticHumanoidObservation:
    """
    Observations that are feasible on real humanoid robots.
    Maps directly to physical sensors.
    """

    @staticmethod
    def get_realistic_obs(mujoco_data, mujoco_model) -> np.ndarray:
        """
        Build observation from realistic sensor readings only.
        """
        # ✅ IMU (Inertial Measurement Unit) — available on all robots
        # Provides orientation and angular velocity of the torso
        torso_orientation = mujoco_data.qpos[3:7]     # quaternion [w,x,y,z]
        torso_angular_vel = mujoco_data.qvel[3:6]     # rad/s

        # ✅ Joint encoders — available on all robots
        # Measure joint angles and velocities
        joint_angles    = mujoco_data.qpos[7:]     # rad
        joint_velocities = mujoco_data.qvel[6:]    # rad/s

        # ✅ Foot contact sensors — available on most humanoids
        # Binary or continuous force measurement
        foot_contacts = mujoco_data.sensordata[:4]  # N (Newtons)

        # ✅ Previous action — always available (we sent it!)
        # Helps the policy model motor dynamics and plan ahead
        # (store this externally and pass in)

        # ❌ Global position (x, y, z) — NOT available without GPS/mocap
        # ❌ Global velocity — NOT directly measurable
        # ❌ Contact point positions — NOT directly measurable
        # ❌ Joint torques (ground truth) — only approximate via current sensing

        obs = np.concatenate([
            torso_orientation,     # 4
            torso_angular_vel,     # 3
            joint_angles,          # n_joints
            joint_velocities,      # n_joints
            foot_contacts,         # 4
        ])

        return obs.astype(np.float32)
```

---

### 10.5 Curriculum Learning

Training a humanoid to walk from scratch is very hard. **Curriculum learning** breaks it into stages:

```python
# curriculum_env.py

class CurriculumWalkingEnv(gym.Env):
    """
    Teaches walking in progressive stages.

    Stage 0: Learn to stand upright
    Stage 1: Learn to take small steps in place
    Stage 2: Learn to walk slowly
    Stage 3: Learn to walk at target speed
    Stage 4: Learn to walk on uneven terrain
    """

    def __init__(self):
        super().__init__()
        self.curriculum_level = 0    # Start at easiest stage
        self.success_threshold = 0.8  # Fraction of episodes to succeed before advancing
        self.recent_successes = []

    def _compute_reward(self, info: dict) -> float:
        """
        Reward function changes based on curriculum level.
        """
        if self.curriculum_level == 0:
            # Stage 0: Just stay upright
            return np.cos(info['torso_tilt'])  # max reward = 1.0 when upright

        elif self.curriculum_level == 1:
            # Stage 1: Stay upright + take a step
            upright = np.cos(info['torso_tilt'])
            step_reward = float(info['took_a_step'])
            return upright + 0.5 * step_reward

        elif self.curriculum_level >= 2:
            # Stage 2+: Full walking reward
            velocity_reward = info['x_velocity']
            upright_reward  = np.cos(info['torso_tilt'])
            energy_penalty  = -0.001 * info['energy_used']
            return velocity_reward + upright_reward + energy_penalty

    def _advance_curriculum(self):
        """
        Check if we should move to the next stage.
        Called at the end of each episode.
        """
        recent_mean_reward = np.mean(self.recent_successes[-20:])  # Last 20 episodes

        if (recent_mean_reward > self.success_threshold and 
            self.curriculum_level < 4):
            self.curriculum_level += 1
            print(f"🎓 Curriculum advanced to level {self.curriculum_level}!")
```

---

### 10.6 From Policy to Real Robot

Once your policy is trained, deploying it looks like this:

```python
# deploy_on_real_robot.py (conceptual)

import numpy as np
from stable_baselines3 import PPO
import time

class RealRobotDeployment:
    """
    Conceptual template for deploying a trained policy on real hardware.
    The specific implementation depends on your robot's SDK.
    """

    def __init__(self, policy_path: str, robot_sdk):
        # Load the trained policy (the neural network)
        self.policy = PPO.load(policy_path)
        self.robot = robot_sdk    # Your robot's control API
        self.control_freq = 50   # Hz — how fast we send commands (must match training!)
        self.dt = 1.0 / self.control_freq   # 0.02 seconds per step

    def get_observation(self) -> np.ndarray:
        """
        Read sensors from real robot hardware.
        This replaces what the simulator computed automatically.
        """
        imu_data     = self.robot.get_imu()           # Orientation, angular velocity
        joint_angles = self.robot.get_joint_angles()  # All joint angles
        joint_vels   = self.robot.get_joint_vels()    # All joint velocities
        foot_forces  = self.robot.get_foot_forces()   # Contact forces

        observation = np.concatenate([
            imu_data,
            joint_angles,
            joint_vels,
            foot_forces,
        ]).astype(np.float32)

        return observation

    def run(self, duration_seconds: float = 10.0):
        """
        Main deployment loop.
        """
        steps = int(duration_seconds / self.dt)

        print(f"Deploying policy for {duration_seconds:.0f} seconds ({steps} steps)...")
        self.robot.engage_motors()   # Turn on motor control

        for step in range(steps):
            step_start = time.time()

            # 1. Read observations from real sensors
            obs = self.get_observation()

            # 2. Query trained policy for action
            action, _states = self.policy.predict(obs, deterministic=True)
            # action is a NumPy array of joint commands

            # 3. Send commands to robot
            self.robot.set_joint_torques(action)

            # 4. Safety check
            if self.robot.is_in_danger():
                print("Safety limit hit! Stopping.")
                break

            # 5. Maintain control frequency
            elapsed = time.time() - step_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.robot.disengage_motors()
        print("Deployment complete.")
```

---

### 10.7 The Complete Learning Roadmap

Now that you've finished this guide, here is your path forward:

```
WHERE YOU ARE NOW                    WHERE YOU'RE GOING
─────────────────                    ──────────────────

✅ Gymnasium API                     → Custom MuJoCo environments (MJCF XML)
✅ RL Loop concepts                  → Advanced reward shaping
✅ Custom environment                → Multi-task learning
✅ PPO basics (SB3)                  → SAC, TD3 for sample efficiency
✅ MuJoCo integration                → Isaac Lab GPU training
                                     → Whole-body control
                                     → Motion imitation (motion capture data)
                                     → Terrain curriculum
                                     → Sim-to-real transfer
                                     → Hierarchical RL
                                     → Real robot deployment
```

---

### 10.8 Recommended Next Resources

| Topic              | Resource                                                                     |
| ------------------ | ---------------------------------------------------------------------------- |
| Isaac Lab          | [isaac-lab.github.io](https://isaac-lab.github.io)                           |
| MuJoCo tutorials   | [mujoco.readthedocs.io](https://mujoco.readthedocs.io)                       |
| SB3 documentation  | [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io) |
| Farama Gymnasium   | [gymnasium.farama.org](https://gymnasium.farama.org)                         |
| RL fundamentals    | Sutton & Barto "Reinforcement Learning: An Introduction"                     |
| Robotics RL papers | Google DeepMind, CMU Locomotion Lab, ETH Zurich RSL                          |

---

### 10.9 Final Summary — The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WHAT YOU NOW KNOW                            │
│                                                                 │
│  Gymnasium = Universal interface between algorithms & worlds    │
│                                                                 │
│  Core loop:                                                     │
│    reset() → [step() × N] → reset() → [step() × N] → ...        │
│                                                                 │
│  Every step():                                                  │
│    obs, reward, terminated, truncated, info = env.step(action)  │
│                                                                 │
│  Spaces:                                                        │
│    Box  → continuous (robotics joints, most RL tasks)           │
│    Discrete → integer choices (games, switches)                 │
│                                                                 │
│  MuJoCo:                                                        │
│    XML model → physics engine → Gymnasium wrapper → your agent  │
│                                                                 │
│  Custom env:                                                    │
│    class MyEnv(gym.Env):                                        │
│        __init__, reset, step, render, close                     │
│                                                                 │
│  Training:                                                      │
│    PPO(env=vec_env).learn(total_timesteps=N)                    │
│                                                                 │
│  Robotics pipeline:                                             │
│    Model → Sim (Isaac Lab) → Reward → PPO → Policy → Transfer   │
└─────────────────────────────────────────────────────────────────┘
```

---

*This document covers Gymnasium from absolute beginner to practical robotics RL. Each concept builds on the last — revisit chapters as your understanding deepens. The best way to learn is to run every code example and modify it.*

*Happy training! 🤖*

# Author

Claude 

# 
