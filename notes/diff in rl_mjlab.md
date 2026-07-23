# Code Changes Report (Post-Commit 1425b15)

This document provides a detailed breakdown of all the new and modified lines of code in the [unitree-rl-mjlab](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab) repository after the commit **`1425b15`**.

---

## Summary of Changes

A moving platform task has been introduced for the Unitree G1 humanoid robot. This includes:
1. **Interactive Platform Viewer**: Added manual velocity controls to the Viser viewer in [scripts/play.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/scripts/play.py).
2. **Moving Platform Model Spec**: Configured a new MuJoCo model spec with slider joints in [src/assets/robots/unitree_g1/xmls/platform_g1.xml](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/platform_g1.xml) and registered it in robot constants.
3. **Observation & Reward Filters**: Adjusted configurations in [src/tasks/velocity/config/g1/env_cfgs.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/config/g1/env_cfgs.py) to prevent the policy from seeing or trying to actuate the platform's slide joints directly.
4. **Platform Dynamics & Relative Rewards**: Implemented custom platform events, curriculum, and relative velocity tracking/slip rewards in [src/tasks/velocity/mdp/platform_events.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/platform_events.py).

---

## Detailed File-by-File Changes

### 1. [scripts/play.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/scripts/play.py)

#### [NEW] [PlatformViserPlayViewer](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/scripts/play.py#L42-L103) (Lines 42–103)
Subclasses `ViserPlayViewer` to add manual platform velocity sliders to the GUI:
```python
class PlatformViserPlayViewer(ViserPlayViewer):
  """Subclass of ViserPlayViewer to add manual platform velocity sliders."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._platform_vx_slider = None
    self._platform_vy_slider = None
    self._platform_global_ctrl_ids = None
    self._robot = None

  def setup(self) -> None:
    super().setup()

    # Access the unwrapped environment and robot entity.
    env = self.env.unwrapped
    robot = env.scene["robot"]
    try:
      # Find local actuator IDs for platform velocities
      actuator_ids, _ = robot.find_actuators(("platform_x_vel", "platform_y_vel"))
      self._robot = robot
      # Get the global ctrl indices compiled into the model
      self._platform_global_ctrl_ids = robot.data.indexing.ctrl_ids[actuator_ids]

      # Add a GUI folder for Platform Control under the "Controls" tab
      with self._server.gui.add_folder("Platform Control"):
        self._platform_vx_slider = self._server.gui.add_slider(
          "Platform Velocity X (m/s)",
          min=-2.0,
          max=2.0,
          step=0.05,
          initial_value=0.0,
        )
        self._platform_vy_slider = self._server.gui.add_slider(
          "Platform Velocity Y (m/s)",
          min=-2.0,
          max=2.0,
          step=0.05,
          initial_value=0.0,
        )

        # Quick zeroing button
        zero_btn = self._server.gui.add_button("Zero Platform Speed")
        @zero_btn.on_click
        def _(_) -> None:
          self._platform_vx_slider.value = 0.0
          self._platform_vy_slider.value = 0.0

      print("[INFO] Registered manual platform controls in Viser viewer.")
    except Exception as e:
      print(f"[INFO] Platform actuators not found or not compiled, skipping platform GUI: {e}")

  def sync_viewer_to_env(self) -> None:
    super().sync_viewer_to_env()
    # Write the slider velocities to the platform's native actuators every step
    if self._platform_global_ctrl_ids is not None and self._robot is not None:
      vx = self._platform_vx_slider.value
      vy = self._platform_vy_slider.value
      device = self._robot.data.data.ctrl.device
      vel = torch.tensor([vx, vy], device=device, dtype=torch.float)
      self._robot.data.data.ctrl[:, self._platform_global_ctrl_ids] = vel
```

#### [MODIFY] Play Viewer registration (Line 235)
Replaced the default `ViserPlayViewer` with `PlatformViserPlayViewer`:
```diff
-    ViserPlayViewer(env, policy).run()
+    PlatformViserPlayViewer(env, policy).run()
```

---

### 2. [src/assets/robots/\_\_init\_\_.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/__init__.py)

#### [MODIFY] Imports (Line 18)
Imported the platform configuration getter:
```python
  get_g1_platform_robot_cfg as get_g1_platform_robot_cfg,
```

---

### 3. [src/assets/robots/unitree_g1/g1_constants.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py)

#### [NEW] [PLATFORM_G1_XML](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py#L26-L29) (Lines 26–29)
Added path definition and assertion for the platform XML file:
```python
PLATFORM_G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "platform_g1.xml"
)
assert PLATFORM_G1_XML.exists()
```

#### [NEW] [get_platform_spec](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py#L44-L47) (Lines 44–47)
Added function to load the platform's MuJoCo spec:
```python
def get_platform_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PLATFORM_G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec
```

#### [NEW] [get_g1_platform_robot_cfg](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py#L298-L305) (Lines 298–305)
Added configuration helper for the G1 robot on the moving platform:
```python
def get_g1_platform_robot_cfg() -> EntityCfg:
  """Get a fresh G1 platform robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_platform_spec,
    articulation=G1_ARTICULATION,
  )
```

---

### 4. [src/assets/robots/unitree_g1/xmls/platform_g1.xml](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/platform_g1.xml)

#### [NEW] Entire XML File (Lines 1–22)
Introduces the moving platform body (using slide joints `platform_x` and `platform_y`) actuated by velocity controllers:
```xml
<mujoco model="platform_g1">
  <compiler angle="radian" meshdir="assets"/>
  <include file="g1.xml"/>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <body name="platform" pos="0 0 -0.01">
      <joint name="platform_x" type="slide" axis="1 0 0" limited="false" damping="50"/>
      <joint name="platform_y" type="slide" axis="0 1 0" limited="false" damping="50"/>
      <geom name="platform" type="box" size="10 10 0.01" material="groundplane"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="platform_x_vel" joint="platform_x" kv="100000"/>
    <velocity name="platform_y_vel" joint="platform_y" kv="100000"/>
  </actuator>
</mujoco>
```

---

### 5. [src/tasks/velocity/config/g1/\_\_init\_\_.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/config/g1/__init__.py)

#### [MODIFY] Import (Line 7)
```python
  unitree_g1_platform_env_cfg,
```

#### [NEW] Task Registration (Lines 26–33)
Registered the new `Unitree-G1-Platform` environment:
```python
register_mjlab_task(
  task_id="Unitree-G1-Platform",
  env_cfg=unitree_g1_platform_env_cfg(),
  play_env_cfg=unitree_g1_platform_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

---

### 6. [src/tasks/velocity/config/g1/env_cfgs.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/config/g1/env_cfgs.py)

#### [MODIFY] Imports (Lines 6, 8–9, 13)
```python
  get_g1_platform_robot_cfg,
# ...
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
# ...
from mjlab.managers.curriculum_manager import CurriculumTermCfg
```

#### [NEW] [unitree_g1_platform_env_cfg](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/config/g1/env_cfgs.py#L204-L363) (Lines 204–363)
Configures moving platform environment dynamics, joint filters, platform-relative event terms, reward terms, and step curriculums:
```python
def unitree_g1_platform_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 moving platform configuration.

  The platform moves with random velocities during training so the robot
  learns to balance on moving surfaces (e.g. bus, train). Velocity tracking
  and foot slip rewards use platform-relative computations.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  import src.tasks.velocity.mdp as src_mdp

  # 1. Load robot config with the platform and enable platform collisions.
  robot_cfg = get_g1_platform_robot_cfg()
  from mjlab.utils.spec_config import CollisionCfg
  platform_collision = CollisionCfg(
    geom_names_expr=(".*_collision", "platform"),
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1, "platform": 3},
    priority={r"^(left|right)_foot[1-7]_collision$": 1, "platform": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,), "platform": (0.6,)},
  )
  robot_cfg.collisions = (platform_collision,)
  cfg.scene.entities = {"robot": robot_cfg}

  # 2. Disable default terrain.
  cfg.scene.terrain = None

  # 3. Update contact sensor pattern to use prefixed platform name.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "feet_ground_contact":
      sensor.secondary.pattern = "robot/platform"

  # 4. Filter actor and critic joint observation spaces back to 29 joints.
  #    (Platform slide joints must not appear in the observation space.)
  for group_name in ["actor", "critic"]:
    group = cfg.observations[group_name]

    if "joint_pos" in group.terms:
      term = group.terms["joint_pos"]
      group.terms["joint_pos"] = ObservationTermCfg(
        func=term.func,
        noise=term.noise,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint",))}
      )

    if "joint_vel" in group.terms:
      term = group.terms["joint_vel"]
      group.terms["joint_vel"] = ObservationTermCfg(
        func=term.func,
        noise=term.noise,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_joint",))}
      )

  # 5. Filter event joint resets to robot joints only.
  if "reset_robot_joints" in cfg.events:
    cfg.events["reset_robot_joints"].params["asset_cfg"].joint_names = (".*_joint",)

  # 6. Filter posture and joint-limit rewards to prevent tracking the platform joints.
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].params["asset_cfg"].joint_names = ".*_joint"
  if "stand_still" in cfg.rewards:
    cfg.rewards["stand_still"].params["asset_cfg"].joint_names = ".*_joint"
  if "joint_pos_limits" in cfg.rewards:
    cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_joint")
  if "joint_acc_l2" in cfg.rewards:
    cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_joint")

  # 7. Explicitly filter action space to robot joints only.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (".*_joint",)

  # ------------------------------------------------------------------
  # 8. Moving platform events.
  # ------------------------------------------------------------------

  # 8a. Step event: smoothly ramp platform velocity toward random targets.
  cfg.events["set_platform_velocity"] = EventTermCfg(
    func=src_mdp.set_platform_velocity,
    mode="step",
    params={
      "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
      "ramp_rate": 0.5,
      "hold_time_s": (2.0, 5.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("platform_x", "platform_y")),
    },
  )

  # 8b. Reset event: zero platform joint position & velocity each episode.
  cfg.events["reset_platform_joints"] = EventTermCfg(
    func=src_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("platform_x", "platform_y")),
    },
  )

  # ------------------------------------------------------------------
  # 9. Platform-relative rewards.
  # ------------------------------------------------------------------

  # 9a. Replace velocity tracking with platform-relative version.
  import math
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=src_mdp.track_linear_velocity_platform_relative,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  )

  # 9b. Replace foot slip with platform-relative version.
  site_names = ("left_foot", "right_foot")
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=src_mdp.feet_slip_platform_relative,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # 9c. Adjust action rate penalty for platform walking.
  if "action_rate_l2" in cfg.rewards:
    cfg.rewards["action_rate_l2"].weight = -0.03

  # ------------------------------------------------------------------
  # 10. Platform velocity curriculum (speed + sharpness).
  # ------------------------------------------------------------------
  cfg.curriculum["platform_velocity"] = CurriculumTermCfg(
    func=src_mdp.platform_velocity_curriculum,
    params={
      "event_name": "set_platform_velocity",
      "velocity_stages": [
        # Stage 0: Gentle, slow platform motion.
        {"step": 10000 * 24,
         "x": (-0.8, 0.8), "y": (-0.8, 0.8),
         "ramp_rate": 5.0},
        # Stage 1 (~5k iters): Medium speed, moderate ramp.
        {"step": 10500 * 24,
         "x": (-1.4, 1.4), "y": (-1.4, 1.4),
         "ramp_rate": 10.0},
        # Stage 2 (~10k iters): Full speed, sharp changes.
        {"step": 12000 * 24,
         "x": (-2.0, 2.0), "y": (-2.0, 2.0),
         "ramp_rate": 20.0},
      ],
    },
  )

  # Apply play mode overrides for the platform.
  if play:
    # Disable platform velocity randomization during play.
    cfg.events.pop("set_platform_velocity", None)
    cfg.curriculum.pop("platform_velocity", None)

  return cfg
```

---

### 7. [src/tasks/velocity/mdp/\_\_init\_\_.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/__init__.py)

#### [MODIFY] Import (Line 8)
Added the relative import to expose the platform events/rewards:
```python
from .platform_events import *  # noqa: F403
```

---

### 8. [src/tasks/velocity/mdp/platform_events.py](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/platform_events.py)

#### [NEW] Entire File (Lines 1–316)
Implements all movement controllers and custom platform rewards. See the full file link above for implementation details.
Key components implemented:
- Class [set_platform_velocity](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/platform_events.py#L51-L154): Controls time-based speed ramping and controller inputs for moving platforms.
- Class [track_linear_velocity_platform_relative](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/platform_events.py#L160-L212): Reward calculation adjusted to subtract the platform's velocity.
- Class [feet_slip_platform_relative](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/platform_events.py#L218-L281): Penalty calculation adjusted to subtract the platform's velocity.
- Function [platform_velocity_curriculum](file:///C:/Users/admin/Desktop/rl-humanoid/unitree_rl_mjlab/src/tasks/velocity/mdp/platform_events.py#L294-L315): Event parameter schedule updater based on current training iteration.
