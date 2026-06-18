from time import sleep
import mujoco
import mujoco.viewer
#import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("assets/scene_29dof.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

#Plot
# plt.ion()
# fig, ax = plt.subplots()
# x_data = []
# y_data = []
# line, = ax.plot(x_data, y_data)
# t = 0

while viewer.is_running():
    mujoco.mj_step(model, data)
    # plot_data = data.qpos[1]
    # x_data.append(t)
    # y_data.append(plot_data)
    # line.set_xdata(x_data)
    # line.set_ydata(y_data)
    # ax.relim()
    # ax.autoscale_view()
    # if t % 100 == 0:
    #     plt.draw()
    #     plt.pause(0.001)
    # t += 1
    # Press Cltr + / to comment/uncomment 
    sleep(0.05)
    viewer.sync()