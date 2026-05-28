from time import sleep
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("pendulum.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

while viewer.is_running():
    mujoco.mj_step(model, data)
    sleep(0.001)
    viewer.sync()