from envs.g1_env import G1Env
import numpy as np
from time import sleep

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

for i in range(5000):
    env.step(np.zeros(29))
    sleep(env.dt)
    env.render()
    #cam
    if i == 0:
        viewer = env.unwrapped.mujoco_renderer.viewer
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.model.body("pelvis").id
    
env.close()