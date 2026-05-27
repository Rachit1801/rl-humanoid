## MuJoCo

Multi-Joint dynamics with Contact (MuJoCo) is a physics simulator.

A virtual physics world where robots can learn and move.

It simulates gravity, friction, collisions, motors, joints, sensors, contact forces

Train a humanoid to walk in MuJoCo then transfer policy to real robot. This is called Sim-to-Real

**Install MuJoCo**

```
pip install mujoco
```

#### MuJoCo Simulation Loop

Create Model -> Create Data -> Loop action    

#### Important MuJoCo Concepts

**Bodies** : Physical objects **Joints** : How bodies move **Geoms** : Collision shapes. **Actuators** : Motors controlling joints.

### MJCF/XML files

#### Format

```
World
 ├── Bodies
 │     ├── Joints
 │     ├── Geoms
 │     └── Sensors
 │
 └── Actuators
```

#### Structure

```xml
<mujoco>
    <worldbody>

    </worldbody>
</mujoco>
```

#### Terms

`<mujoco>` : Root tag

`<worldbody>`  : Contains all physical things in the world. Example floor, robot, walls, objects

`<geom>` : A geom is visible shape, collision shape, physical object. Example box, sphere, capsule, cylinder, plane

`size="5 5 2"`  : Means half length x,y and z. $10 × 10 × 4$

`<body>` : A body is a movable rigid body

Coordinate System : x = right, y = forward/back, z = up

`<freejoint/>` : This gives the body full 6 DOF movement. Translate x/y/z and rotate x/y/z. Without it body becomes fixed in air.

`rgba="1 0 0 1"` : Values between 0 and 1

**Note :**

> A body itself is invisible, only geoms are visible.
> 
> Bodies can contain bodies.

#### Example : Box falling

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

#### Joints
