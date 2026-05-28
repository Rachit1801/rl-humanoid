# MuJoCo

Multi-Joint dynamics with Contact (MuJoCo) is a physics simulator.

A virtual physics world where robots can learn and move.

It simulates gravity, friction, collisions, motors, joints, sensors, contact forces

Train a humanoid to walk in MuJoCo then transfer policy to real robot. This is called Sim-to-Real

**Install MuJoCo**

```
pip install mujoco
```

## MuJoCo Simulation Loop

Create Model -> Create Data -> Loop action    

## Important MuJoCo Concepts

**Bodies** : Physical objects **Joints** : How bodies move **Geoms** : Collision shapes. **Actuators** : Motors controlling joints.

## MJCF/XML files

### Format

```
World
 ├── Bodies
 │     ├── Joints
 │     ├── Geoms
 │     └── Sensors
 │
 └── Actuators
```

### Structure

```xml
<mujoco>
    <worldbody>

    </worldbody>
    <actuator>

    </actuator>
    <sensor>
        
    </sensor>
</mujoco>
```

### Terms

- `<mujoco>` : Root tag

- `<worldbody>`  : Contains all physical things in the world. Example floor, robot, walls, objects

- `<geom>` : A geom is visible shape, collision shape, physical object. Example box, sphere, capsule, cylinder, plane

- `size="5 5 2"`  : Means half length x,y and z. $10 × 10 × 4$

- `<body>` : A body is a movable rigid body
  
  - **Coordinate System** : x = right, y = forward/back, z = up

- `<freejoint/>` : This gives the body full 6 DOF movement. Translate x/y/z and rotate x/y/z. Without it body becomes fixed in air.

- `rgba="1 0 0 1"` : Values between 0 and 1

**Note :**

> A body itself is invisible, only geoms are visible.
> 
> Bodies can contain bodies.

### Example : Box falling

```python
import mujoco
import mujoco.viewer

xml = """
<mujoco>
    <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml) #Compiles the XML world.
data = mujoco.MjData(model) #Stores pos, velocity, force, state

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data) #Advances physics by one timestep.
        viewer.sync()
```

### Joints

**Structure**

```xml
<body>
    <joint/>
</body>
```

#### Types of Joints

| Joint | Motion                   |
| ----- | ------------------------ |
| free  | move + rotate freely     |
| hinge | rotate                   |
| slide | move linearly            |
| ball  | rotate in all directions |

`axis="x y z"` 

Joint motion is relative to parent body

##### Hinge

Example : a long bar connected by hinge gravity rotates it. Like a falling stick.

```xml
 <body pos="0 0 1">
     <joint type="hinge" axis="0 1 0"/>
     <geom type="box" size="0.5 0.1 0.1"/>
 </body>
```

##### Slide Joint

Linear movement Example: piston, elevator

`<joint type="slide" axis="1 0 0"/>` allows translation along x-axis.

##### Free Joint

`<freejoint/>`

3 translations  + 3 rotations

<mark>Only ONE freejoint per kinematic tree root. </mark>Humanoids usually have torso as freejoint, everything else is hinge joints

##### Capsule

Used because stable collisions, smooth edges

**Fromto :** `fromto="x1 y1 z1 x2 y2 z2"` defines start point and end point. 
Example: `fromto="0 0 0 1 0 0"` capsule goes from x=0 to x=1

#### Parent-Child Bodies

```xml
<mujoco>
    <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <!-- Parent -->
        <body pos="0 0 0.5">
            <joint type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.5 0.5 0.5" rgba="1 0 0 1"/>
            <!-- Child -->
            <body pos="0 0 0.5">
                <joint type="hinge" axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 2" size="0.1" rgba="0 1 0 1"/>
            </body>
        </body>
    </worldbody>
</mujoco>
```

Child body has `pos="0 0 0.5"`, means child body starts at end of parent. This creates connected links. 

The child body position is relative to parent. NOT world coordinates.

---

### Actuators

apply forces to joints

Actuators are written OUTSIDE `worldbody`

#### Types

| Type     | Purpose             |
| -------- | ------------------- |
| motor    | direct torque/force |
| position | PD position control |
| velocity | velocity control    |

#### Example

```xml
<actuator>
        <motor joint="cart_slider" ctrlrange="-1 1" gear="100"/>
</actuator>
```

It is important to give joint names because actuators need to know which joint to control 

#### Terms

- <actuator> : Contains motors/controllers.

- <motor> : apply force to this joint

- `ctrlrange="-1 1"` : Allowed control input. RL actions become: -1 ≤ action ≤ 1. 
  -1 = push left, +1 = push right.

- `gear="100"`: Motor strength multiplier/torque multiplier.

#### Control

PPO outputs action = 0.7 then MuJoCo actuator converts this into force, torque and motion. The agent DOES NOT know physics, joints and motors. It only observes, does action and gives reward

Each actuator receives a value `data.ctrl[i]` this is how Python controls robot.

Example : cart moves automatically

```python
while viewer.is_running():
    data.ctrl[0] = 0.5
    mujoco.mj_step(model, data)
    viewer.sync()
```

### Sensors

Sensors are defined outside `worldbody`.

#### Types

| Sensor        | Measures         |
| ------------- | ---------------- |
| jointpos      | joint angle      |
| jointvel      | joint speed      |
| accelerometer | acceleration     |
| gyro          | angular velocity |
| force         | forces           |
| touch         | contacts         |

#### Example

Creates sensors for pole angle and pole angular velocity

```xml
<sensor>
    <jointpos joint="pole_hinge"/>
    <jointvel joint="pole_hinge"/>
</sensor>
```

#### Reading Sensor Data

```python
print(data.sensordata)
```

## Observations

An RL agent cannot see the world directly. It only receives numbers describing the state. These numbers are called observations or state.

#### qpos

Joint Positions 
Contains joint angles, positions and orientations

Example: `qpos = [0.5, 0.2]` Means cart at x=0.5 and pole angle=0.2 rad

#### qvel

Joint Velocities

Example: `qvel = [1.2, -0.7]` Means cart moving right and pole rotating left

#### Example

```python
print(data.qpos)
print(data.qvel)
```

Every joint adds position and velocity to the observation space.

Gymnasium usually combines these into `obs = np.concatenate([ qpos, qvel, sensors` . This becomes input to PPO.
