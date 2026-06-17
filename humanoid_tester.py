from envs.g1_env import G1Env
import numpy as np
from time import sleep

import types
import mujoco

"""
Purpose of this file is to test if the bot is able to retain its initial position. 
The script removes PPO completely
If the robot stands, your controller is working and you can start RL.
If it falls, you fix the controller first because PPO cannot learn balance
on top of a broken standing controller.
"""

env = G1Env(render_mode="human")
obs, _ = env.reset()

platform_pos = np.array([0.0, 0.0, -0.05])
platform_vel = np.array([0.1, 0.0, 0.0])
platform_acc = np.array([0.05, 0.0, 0.0])

def custom_do_simulation(self, ctrl, n_frames):
    global platform_pos, platform_vel
    self.data.ctrl[:] = ctrl
    for _ in range(n_frames):
        platform_vel += platform_acc * self.model.opt.timestep
        platform_pos += platform_vel * self.model.opt.timestep
        self.data.mocap_pos[0] = platform_pos

        mujoco.mj_step(self.model, self.data)

env.do_simulation = types.MethodType(custom_do_simulation, env)
dt = env.dt

for i in range(5000):
    env.step(np.zeros(29))
    sleep(dt)
    env.render()
    #cam
    if i == 0:
        viewer = env.unwrapped.mujoco_renderer.viewer
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("pelvis").id
    
env.close()