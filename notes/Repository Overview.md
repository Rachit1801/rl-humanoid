## Repository Overview

### What This Repository Does

**unitree_rl_mjlab** is an end-to-end reinforcement learning framework for training locomotion policies on **Unitree robots** (quadrupeds and humanoids) using **MuJoCo** as the physics engine. It spans the full pipeline from simulation training to real-robot deployment.

Supported robots: Go2, A2, As2, G1 (29 & 23 DOF), R1, H1, H2.

Two task paradigms are supported:

1. **Velocity Tracking** — the robot follows commanded linear and angular velocities
2. **Motion Imitation (Tracking)** — the robot reproduces reference motion-capture sequences (e.g., dance)

### Architecture

The system is split into **two language ecosystems** that communicate through a shared ONNX model artifact:

![system](C:\Users\admin\Desktop\system.svg)

**Framework Delegation** — The repo itself contains no RL algorithms, no physics stepping, and no environment loop. It delegates entirely to:

- **mjlab** for environment orchestration (Isaac Lab API patterns)
- **rsl_rl** for PPO training loop
- **MuJoCo / MuJoCo Warp** for physics

##  Folder Tree

```
unitree_rl_mjlab/
│
├── scripts/                    # CLI entry points (Python)
│
├── src/                        # Python package — task & asset definitions
│   ├── assets/
│   │   ├── robots/             # Robot articulation configs + MJCF models
│   │   └── motions/            # Motion-capture reference data (CSV, NPZ)
│   │
│   └── tasks/
│       ├── velocity/           # Velocity tracking task
│       │   ├── config/         #   Per-robot env configurations
│       │   ├── mdp/            #   Reward functions & observations
│       │   └── rl/             #   RL agent (PPO) hyperparameters
│       │
│       └── tracking/           # Motion imitation task
│           ├── config/         #   Per-robot env configurations
│           ├── mdp/            #   Reward & observation functions
│           └── rl/             #   RL agent (PPO) hyperparameters
│
├── deploy/                     # C++ real-robot deployment
│   ├── include/                # Shared headers (FSM, articulation, joystick)
│   │   ├── FSM/                #   Finite state machine framework
│   │   └── isaaclab/           #   Ported Isaac Lab utilities (C++)
│   ├── robots/                 # Per-robot deployment packages
│   │   ├── g1/                 #   G1-specific controller, configs, ONNX
│   │   ├── go2/                #   Go2-specific controller
│   │   ├── h1_2/               #   H1_2-specific controller
│   │   └── ...                 #   (a2, g1_23dof, r1)
│   └── thirdparty/             # Vendored deps (ONNX Runtime, cnpy)
│
├── simulate/                   # MuJoCo simulation bridge (C++)
│   ├── mujoco/                 # MuJoCo binaries, models, samples
│   └── src/                    # Sim bridge source (SDK2 integration)
│
├── doc/                        # Documentation and demo GIFs
├── setup.py                    # Python package installer
└── README.md                   # Project documentation
```

------

### Folder Details

------

#### `scripts/`

> Single collection of all user-facing CLI entry points.

| File                 | Purpose                                                      |
| :------------------- | :----------------------------------------------------------- |
| train.py             | Main training loop — parses CLI args, creates mjlab env + rsl_rl runner, runs PPO, exports ONNX |
| play.py              | Policy evaluation — loads a `.pt` checkpoint, replays in MuJoCo viewer, optionally records video |
| csv_to_npz.py        | Motion preprocessing — converts CSV motion-capture files to NPZ format with FPS resampling and body mapping |
| list_envs.py         | Utility — lists all registered Gymnasium environments from this package |
| visualize_terrain.py | Utility — generates and visualizes procedural terrain heightmaps (stairs, slopes, rough ground) |

**Typical responsibilities**: Argument parsing, environment construction, training orchestration, model I/O. These scripts **do not** contain RL algorithm code — they delegate to `mjlab` and `rsl_rl`.

------

#### `src/assets/robots/`

> Defines the physical properties and articulation structure for every supported robot.

Each robot has its own subdirectory (e.g., `unitree_g1/`, `unitree_go2/`) containing:

| Contents             | Role                                                         |
| :------------------- | :----------------------------------------------------------- |
| `__init__.py`        | Exports `ArticulationCfg` constants (e.g., `UNITREE_G1_CFG`) |
| `g1.py` (or similar) | Defines the `ArticulationCfg` dataclass — MJCF path, joint ordering, actuator gains (stiffness/damping), default joint positions |
| `*.xml` / MJCF files | MuJoCo model definitions (mesh references, body hierarchy, sensors) |

The top-level robots/__init__.py re-exports all robot configs so tasks can import them by name (e.g., `from src.assets.robots import UNITREE_G1_CFG`).

**Key pattern**: Actuators use `ImplicitActuatorCfg` with per-joint PD gains (`stiffness`, `damping`) — these model the low-level motor controllers on the physical robots.

------

#### `src/assets/motions/`

> Stores motion-capture reference data used by the motion imitation (tracking) task.

| Contents           | Role                                                         |
| :----------------- | :----------------------------------------------------------- |
| `g1/`, `g1_23dof/` | Robot-specific motion directories                            |
| `*.csv`            | Raw motion capture at source FPS                             |
| `*.npz`            | Processed motion data (generated by `csv_to_npz.py`) — contains joint angles, root positions/orientations, velocities at target simulation FPS |

------

#### `src/tasks/velocity/`

> Defines the complete **velocity tracking** RL task — environment configuration, MDP components, and RL hyperparameters.

| Subdirectory / File | Role                                                         |
| :------------------ | :----------------------------------------------------------- |
| velocity_env_cfg.py | **Base environment config** — `UnitreeVelocityFlatEnvCfg` dataclass defining scene, observations, actions, rewards, terminations, curriculum, commands. All robot-specific configs inherit from this. |
| `config/`           | **Per-robot overrides** — e.g., `g1/flat_env_cfg.py` creates `UnitreeG1FlatEnvCfg` by plugging in the G1 articulation, adjusting joint limits, reward weights, and action scales. Each config file also calls `gymnasium.register()` to make the task available by name. |
| `mdp/`              | **MDP building blocks** — custom reward functions (e.g., `track_lin_vel_xy_exp`, `feet_air_time`, `joint_power_penalty`) and custom observation terms (e.g., filtered base angular velocity) |
| `rl/`               | **Agent config** — PPO hyperparameters, policy network architecture (MLP sizes), learning rates, GAE parameters, typically as `RslRlPpoActorCriticCfg` |

**Reward architecture** (velocity task): A weighted sum of ~15-20 terms including:

- Velocity tracking (linear XY, angular Z) — primary objective
- Regularization (joint torques, action rate, smoothness)
- Gait shaping (feet air time, base height, orientation)
- Safety (joint limits, collision penalties)

------

#### `src/tasks/tracking/`

> Defines the **motion imitation** RL task — tracks reference motion sequences rather than velocity commands.

The structure mirrors `velocity/` exactly but with key differences:

| Aspect           | Velocity                           | Tracking                                                     |
| :--------------- | :--------------------------------- | :----------------------------------------------------------- |
| **Commands**     | Linear/angular velocity            | Reference motion trajectory                                  |
| **Observations** | Base state + commands              | Base state + reference joint/body targets                    |
| **Rewards**      | Velocity tracking + regularization | Joint angle tracking + body position/orientation tracking + style rewards |
| **Config base**  | `UnitreeVelocityFlatEnvCfg`        | `UnitreeTrackingEnvCfg`                                      |

The tracking MDP loads NPZ motion files and provides per-timestep reference poses as part of the observation vector, enabling the policy to learn imitation.

------

#### `deploy/`

> C++ real-time controller for deploying trained policies on physical Unitree robots.

##### `deploy/include/FSM/`

A **Finite State Machine** framework that manages robot operating modes:

![FSM](C:\Users\admin\Desktop\FSM.svg)

------

#### `simulate/`

> A MuJoCo-based simulation bridge that mimics the real robot's SDK interface, enabling **sim deployment testing** before touching hardware.

| File                  | Role                                                         |
| :-------------------- | :----------------------------------------------------------- |
| main.cc               | Entry point — loads MJCF model, launches MuJoCo viewer, runs physics loop |
| unitree_sdk2_bridge.h | **Key component** — implements a DDS bridge that exposes MuJoCo simulation as if it were a real Unitree robot, so the deploy controller can connect via `--network=lo` |
| physics_joystick.h    | Gamepad input handling for commanding the simulated robot    |
| param.h               | Simulation parameters (timestep, model paths)                |
| config.yaml           | Robot selection and scene configuration                      |
| CMakeLists.txt        | Build config linking MuJoCo, CycloneDDS, Unitree SDK2        |

**Architecture insight**: The simulate bridge uses CycloneDDS to publish simulated sensor data (joint states, IMU) and subscribe to motor commands — **the same protocol** the real robot uses. This means the deploy controller binary runs identically against simulation and hardware, differing only by the network interface argument (`lo` vs `enp5s0`).

------

### Data Flow Summary

### ![1flow](C:\Users\admin\Desktop\1flow.svg)

------

### Key External Dependencies

| Dependency                                                   | Version     | Role                                              |
| :----------------------------------------------------------- | :---------- | :------------------------------------------------ |
| [mjlab](https://github.com/mujocolab/mjlab)                  | 1.2.0       | Isaac Lab-style environment framework over MuJoCo |
| [mujoco-warp](https://github.com/google-deepmind/mujoco_warp) | 3.5.0       | GPU-accelerated MuJoCo physics                    |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl)           | (via mjlab) | PPO algorithm, actor-critic networks              |
| [ONNX Runtime](https://onnxruntime.ai/)                      | 1.22.0      | C++ neural network inference                      |
| [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) | —           | Robot communication (DDS)                         |
| [CycloneDDS](https://github.com/eclipse-cyclonedds/cyclonedds) | —           | DDS middleware for robot comms                    |

## File Descriptions

### train.py

- **Purpose**: Acts as the main training orchestrator for reinforcement learning tasks.
- **Main Classes**: `TrainConfig` (frozen dataclass).
- **Main Functions**: `run_train()`, `launch_training()`, `main()`.
- **Inputs**: Command-line arguments specifying the target task (e.g., `"Unitree-G1-Flat"`) and overrides for environment/agent parameters.
- **Outputs**: Policy model files (`model_N.pt`) and inference ONNX bundles (`policy.onnx` + `policy.onnx.data`).
- **Dependencies**: `tyro`, `torch`, `mjlab`, `rsl_rl`, `src.tasks`.
- **Who imports it**: None (executed directly as a script).
- **Role in Training Pipeline**: Sets up the execution context, spawns environments, and triggers the PPO optimization loop.
- **Typical execution order**: Executed at the very start of the workflow.

### play.py

- **Purpose**: Replays and visualizes trained policies inside native or web-based MuJoCo viewers.
- **Main Classes**: `PlayConfig` (frozen dataclass), `PolicyZero`, `PolicyRandom`.
- **Main Functions**: `run_play()`, `main()`.
- **Inputs**: Command line argument specifying the target task, and paths to the saved `.pt` checkpoint file.
- **Outputs**: Interactive rendering of the robot's physical behavior, or recorded video files.
- **Dependencies**: `tyro`, `torch`, `mjlab`, `src.tasks`.
- **Who imports it**: None (executed directly as a script).
- **Role in Training Pipeline**: Post-training evaluation and validation step.
- **Typical execution order**: Run after training is complete to assess policy performance.

### velocity_env_cfg.py

- **Purpose**: Formulates the general environment blueprints for velocity tracking tasks.
- **Main Classes**: `UnitreeVelocityFlatEnvCfg` (inherits `ManagerBasedRlEnvCfg`).
- **Main Functions**: `make_velocity_env_cfg()`.
- **Inputs**: None.
- **Outputs**: A populated base `ManagerBasedRlEnvCfg` object.
- **Dependencies**: `mjlab`, `src.tasks.velocity.mdp`.
- **Who imports it**: Per-robot configuration files (e.g., `src/tasks/velocity/config/g1/env_cfgs.py`).
- **Role in Training Pipeline**: Supplies the base settings for sensors, default rewards, observation patterns, and domain randomizations.
- **Typical execution order**: Loaded dynamically during environment initialization.

### env_cfgs.py (Velocity - G1)

- **Purpose**: Customizes the base velocity-tracking configuration specifically for the G1 humanoid.
- **Main Classes**: `UnitreeG1FlatEnvCfg` (inherits `UnitreeVelocityFlatEnvCfg`).
- **Main Functions**: `register_g1_flat_env()`, `register_g1_rough_env()`.
- **Inputs**: None.
- **Outputs**: Gym-registered environment IDs.
- **Dependencies**: `gymnasium`, `mjlab`, `src.tasks.velocity.velocity_env_cfg`, `src.assets.robots`.
- **Who imports it**: Auto-discovered by `src/tasks/__init__.py`.
- **Role in Training Pipeline**: Pairs the general velocity task logic with the G1 robot's joints, collision geometries, and action scaling.
- **Typical execution order**: Executed during registry initialization before environments are instantiated.

### g1_constants.py

- **Purpose**: Defines G1 motor specifications, physical limits, and XML paths.
- **Main Classes**: None.
- **Main Functions**: `get_g1_robot_cfg()`, `reflected_inertia_from_two_stage_planetary()`.
- **Inputs**: None.
- **Outputs**: `EntityCfg` (robot configuration object).
- **Dependencies**: `mjlab`, `mujoco`.
- **Who imports it**: `src/assets/robots/__init__.py`.
- **Role in Training Pipeline**: Translates raw physical parameters (such as gear ratios and joint friction) into settings for the MuJoCo simulator.
- **Typical execution order**: Consulted during environment compilation.

---

The files involved in training G1 to walk are:

Orchestration & Entry Points

- scripts/train.py — The CLI script containing the training loop, environment setup, wrapping, and saving processes.

Task Configuration & Registry

- src/tasks/velocity/velocity_env_cfg.py — The base blueprint for velocity tracking tasks, setting default observation blocks, rewards, and environment steps.
- src/tasks/velocity/config/g1/env_cfgs.py — Customizes the base env config for G1 (assigning joint configurations, contact sensors, biped phase parameters, and action scaling).
- src/tasks/velocity/config/g1/rl_cfgs.py — Defines the specific PPO hyperparameters (network sizes, learning rate, entropy coefficient) for G1.

MDP (Markov Decision Process) Components

- src/tasks/velocity/mdp/**init**.py — Re-exports velocity tracking environment MDP terms.
- src/tasks/velocity/mdp/velocity_command.py — Handles the generation of random/commanded translation and heading velocities.
- src/tasks/velocity/mdp/rewards.py — Contains velocity tracking rewards, posture regularizations, foot clearance, and foot slippage penalties.
- src/tasks/velocity/mdp/observations.py — Implements features like the biped cyclic phase gait clock, foot airtimes, and contact states.
- src/tasks/velocity/mdp/terminations.py — Rules for environment resets when the robot falls over or touches the ground with illegal links (torso/arms).
- src/tasks/velocity/mdp/curriculums.py — Controls the progression of terrain levels and command bounds as the policy learns.

Policy Save & ONNX Export Customizations

- src/tasks/velocity/rl/**init**.py — Overrides the RSL-RL runner to export `policy.onnx` with deployment metadata parameters when a checkpoint is saved.

G1 Robot Model & Physical Constants (Assets)

- src/assets/robots/**init**.py — Aggregates and re-exports all robot config utilities.
- src/assets/robots/unitree_g1/**init**.py — G1 asset subpackage namespace initialization.
- src/assets/robots/unitree_g1/g1_constants.py — Sets default joint standing positions, motor limits, stiffness/damping gains, and actuator groups.
- src/assets/robots/unitree_g1/xmls/g1.xml — The MuJoCo XML description file defining kinematics, sensor sites, collision bodies, and actuator joints.
- **Mesh Assets**: Mesh files in src/assets/robots/unitree_g1/xmls/assets/ represent the physical shell structure of the robot and are read directly by the MuJoCo compiler during initialization.
