import mujoco
import mujoco.viewer

xml = """
<mujoco>
    <worldbody>
        <!-- Ground -->
        <geom type="plane" size="5 5 0.1"/>
        <!-- Rail (For Refrence)-->
        <geom type="box" pos="0 0 0" size="2 0.05 0.05" rgba="0.3 0.3 0.3 1"/>
        <!-- Cart -->
        <body pos="0 0 0.1">
            <!-- Cart moves left/right -->
            <joint name="cart_slider" type="slide" axis="1 0 0"/>
            <!-- Cart body -->
            <geom type="box" size="0.2 0.15 0.1" rgba="1 0 0 1"/>
            <!-- Pole -->
            <body pos="0 0 0.1">
                <!-- Pole rotates -->
                <joint type="hinge" axis="0 1 0"/>
                <!-- Pole geom -->
                <geom type="capsule" fromto="0 0 0 0 0 1" size="0.05" rgba="0 1 0 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor joint="cart_slider" ctrlrange="-1 1" gear="1000"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()