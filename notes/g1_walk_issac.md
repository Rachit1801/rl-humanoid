# Code

Source : https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py

velocity_env_cfg.py

```python
import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

```

# Explaination

## ManagerBasedRLEnvCfg

IsaacLab's `ManagerBasedRLEnvCfg` splits that single function into **named managers**, each configured as its own dataclass:

| IsaacLab manager  | Role                                | MuJoCo equivalent                    |
| ----------------- | ----------------------------------- | ------------------------------------ |
| `SceneCfg`        | world: terrain, robot, sensors      | XML/MJCF scene                       |
| `EventCfg`        | domain randomization & resets       | noise                                |
| `CommandsCfg`     | what the robot is told to do        | target                               |
| `ActionsCfg`      | policy output → joint targets       | `action * scale + default_qpos` line |
| `ObservationsCfg` | what the policy sees                | `obs` vector builder                 |
| `RewardsCfg`      | scalar reward                       | reward function                      |
| `TerminationsCfg` | episode-end logic                   | `done` checks                        |
| `CurriculumCfg`   | difficulty scheduling over training |                                      |

## Terrain generation

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(...)
```

Procedurally builds a grid of terrain tiles `num_rows=9 × num_cols=21` = 189 sub-terrains, each `8m × 8m`, surrounded by a `20m` border. `difficulty_range=(0.0, 1.0)` is meant to later hold rough/uneven sub-terrains at increasing difficulty per row currently it's just flat tiles repeated.

## `RobotSceneCfg` : the world

```python
class RobotSceneCfg(InteractiveSceneCfg):
```

| Component                             | What it does                                                 |
| ------------------------------------- | ------------------------------------------------------------ |
| `terrain` (`TerrainImporterCfg`)      | Spawns the terrain above, with `friction_combine_mode="multiply"`, `static/dynamic_friction=1.0` material, and a marble-tile *visual* material |
| `robot` (`ROBOT_CFG.replace(...)`)    | Instantiates the Unitree G1 29-DOF articulation per-env, under `{ENV_REGEX_NS}/Robot` (IsaacLab's per-environment namespace) |
| `height_scanner` (`RayCasterCfg`)     | A grid of downward rays (`1.6m × 1.0m`, `0.1m` resolution) cast from `20m` above the torso, used to perceive terrain height under the robot |
| `contact_forces` (`ContactSensorCfg`) | Tracks per-body contact forces + air time, 3-step history, on *every* body |
| `sky_light`                           | A dome light for rendering                                   |

## `EventCfg`

Randomize 

| Event                        | Use                                                          |
| ---------------------------- | ------------------------------------------------------------ |
| `physics_material`           | Prevents the policy from overfitting to one exact value      |
| `add_base_mass`              | Real robots have payload mass variance; this forces robustness to mass uncertainty |
| `base_external_force_torque` | Infrastructure for applying random disturbance forces at reset |
| `reset_base`                 | Without this every episode would start from the *exact* same pose and policy memorizes one starting configuration instead of learning a general controller |
| `reset_robot_joints`         | Random joint velocities at reset simulate the robot mid-motion rather than always starting from rest |
| `push_robot`                 | This forces the policy to learn active balance recovery, not just steady-state walking |

## `CommandsCfg`

what the robot is told to do

```python
base_velocity = mdp.UniformLevelVelocityCommandCfg(...)
```

Every `10.0s` (`resampling_time_range=(10.0,10.0)`), each environment samples a new target `(lin_vel_x, lin_vel_y, ang_vel_z)` command "walk forward at 0.3 m/s while turning slightly,"

`rel_standing_envs=0.02` means 2% of environments get a zero-velocity so the policy doesn't forget how to balance in place.

## `ActionsCfg` 

```python
JointPositionAction = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
)
```

The actor network outputs one number per joint (29 joints here). This action is transformed into a desired joint position via `joint_target = default_joint_pos + action * 0.25`  

**Why `scale=0.25` and `use_default_offset=True` matter:**

- `use_default_offset=True` means actions are *deltas around the standing pose*, not absolute joint angles. This makes learning vastly easier as the policy starts already in the default pose and just needs to learn small corrections, rather than learning the entire standing pose from scratch via raw angle outputs.
- `scale=0.25` clips how far a single network output of `±1.0` can move a joint from that default preventing violent, unrealistic joint commands during early random exploration, which would otherwise destroy the robot's balance or break PD-controller stability before any useful gradient is collected.

## `ObservationsCfg` 

| Term                | Why                                                          |
| ------------------- | ------------------------------------------------------------ |
| `base_ang_vel`      | IMU gyro reading                                             |
| `projected_gravity` | Gravity vector in body frame                                 |
| `velocity_commands` | The target velocity from `CommandsCfg`                       |
| `joint_pos_rel`     | Joint angles relative to default : proprioception            |
| `joint_vel_rel`     | Joint velocities                                             |
| `last_action`       | Previous action : gives the policy short-term memory and helps in smoothness |

The actual policy input is 5 stacked timesteps of this vector, giving it short-term temporal context (a crude substitute for recurrence).

`history_length=5`

`enable_corruption=True` : turns noise on

`concatenate_terms=True` : flattens into one vector

**`CriticCfg`** : what the *value function* sees

## `RewardsCfg` 

### Task rewards

| Term               | Weight  | Purpose                                                |
| ------------------ | ------- | ------------------------------------------------------ |
| `track_lin_vel_xy` | `+1.0`  | Exponential reward for matching commanded x/y velocity |
| `track_ang_vel_z`  | `+0.5`  | for commanded turning rate                             |
| `alive`            | `+0.15` | Flat bonus every step the episode hasn't terminated    |

### Base stabilization

penalties keeping the torso well-behaved

| Term                                                   | Weight  | Purpose                                                    |
| ------------------------------------------------------ | ------- | ---------------------------------------------------------- |
| `base_linear_velocity` (`lin_vel_z_l2`)                | `-2.0`  | Penalizes bounces/hops                                     |
| `base_angular_velocity` (`ang_vel_xy_l2`)              | `-0.05` | Penalizes wobbling side-to-side and front-back             |
| `flat_orientation_l2`                                  | `-5.0`  | Penalizes leaned/crouched gait and promots upright posture |
| `base_height` (`base_height_l2`, `target_height=0.78`) | `-10`   | Penalizes crouch-walk or over-extend                       |

### Smoothness / energy / hardware-safety penalties

 these matter a lot for sim-to-real and match concerns you'd have had tuning PD gains in MuJoCo

| Term                                  | Weight    | Purpose                                             |
| ------------------------------------- | --------- | --------------------------------------------------- |
| `joint_vel` (`joint_vel_l2`)          | `-0.001`  | Discourages unnecessarily fast joints               |
| `joint_acc` (`joint_acc_l2`)          | `-2.5e-7` | Discourages high joint accelerations                |
| `action_rate` (`action_rate_l2`)      | `-0.05`   | Penalizes large step-to-step changes in action      |
| `dof_pos_limits` (`joint_pos_limits`) | `-5.0`    | Penalizes approaching joint angle mechanical limits |
| `energy`                              | `-2e-5`   | Penalizes total power use                           |

### Posture shaping

 kept-still joints to encourage a natural humanoid gait rather than an alien one

If you removed these three the policy would still walk (task reward dominates) but likely with much more flailing arms, twisted torso, or splayed-out legs physically valid solutions PPO will happily find if nothing discourages them.

| Term                     | Weight | Why                                                  |
| ------------------------ | ------ | ---------------------------------------------------- |
| `joint_deviation_arms`   | `-0.1` | Penalty for arm movement - shoulders, elbows, wrists |
| `joint_deviation_waists` | `-1`   | Discourages waist twisting                           |
| `joint_deviation_legs`   | `-1.0` | Strongly discourages a wide stance - hip roll/yaw    |

### Feet-specific rewards

| Term                                       | Weight | What it does                                                 |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| `gait` (`feet_gait`)                       | `+0.5` | Rewards an alternating left/right contact pattern            |
| `feet_slide`                               | `-0.2` | Penalizes feet moving while in contact with the ground       |
| `feet_clearance` (`foot_clearance_reward`) | `+1.0` | Rewards lifting the swing foot to roughly `target_height=0.1m` |

### Safety/contact penalty

| Term                 | Weight | Why                                                          |
| -------------------- | ------ | ------------------------------------------------------------ |
| `undesired_contacts` | `-1`   | Stops the robot from walking on its knees, torso, or hands touching the ground and forces ground contact to happen only through the feet |

## `TerminationsCfg`

```python
time_out = DoneTerm(func=mdp.time_out, time_out=True)
base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
```

| Term              | Trigger                                     | Why                                                |
| ----------------- | ------------------------------------------- | -------------------------------------------------- |
| `time_out`        | `episode_length_s` elapsed (20s, set later) | Episode length so PPO gets bounded-length rollouts |
| `base_height`     | torso height `< 0.2m`                       | Catches a fall/collapse early                      |
| `bad_orientation` | tilt `> 0.8` rad (~46°) from upright        | Catches a robot that's toppled sideways            |

## `CurriculumCfg`

```python
terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
```

**`terrain_levels_vel`**: tracks, per environment, whether the robot successfully tracked its velocity command over the episode. Robots that succeed get promoted to a harder terrain row next reset. Robots that fail get demoted toward easier rows. Over training, the population naturally spreads across difficulty levels matched to current competence.

**`lin_vel_cmd_levels`**: Widens `CommandsCfg.ranges` toward `limit_ranges` and grows the commanded velocity from the tiny `(-0.1, 0.1)` starting range up toward the full `(-0.5, 1.0)` ceiling.

## `RobotEnvCfg` 

python

```python
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    ...
```

This class just instantiates one of each manager you've now seen, then `__post_init__` wires up timing and consistency:

| Setting                                                    | Value                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| `decimation = 4`                                           | Policy/action runs once every 4 physics steps                |
| `episode_length_s = 20.0`                                  | 20-second episodes                                           |
| `sim.dt = 0.005`                                           | 200Hz physics step                                           |
| `sim.render_interval = decimation`                         | Render only on control steps                                 |
| `sim.physics_material = scene.terrain.physics_material`    | Ground inherits the randomized friction material             |
| `scene.contact_forces.update_period = sim.dt`              | Contact sensor ticks every physics step                      |
| `scene.height_scanner.update_period = decimation * sim.dt` | Height scan ticks once per control step                      |
| curriculum toggle block                                    | If `curriculum.terrain_levels` exists, enable `terrain_generator.curriculum = True`, else `False` |

## `RobotPlayEnvCfg`

The evaluation/deployment config. This is what to load to watch a trained policy

```python
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
```

---

