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

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()