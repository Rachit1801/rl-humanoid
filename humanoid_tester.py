from envs.g1_env_push import G1Env
import numpy as np
from time import sleep

"""
Purpose of this file is to test if the bot is able to retain its initial position. 
The script removes PPO completely
If the robot stands, your controller is working and you can start RL.
If it falls, you fix the controller first because PPO cannot learn balance
on top of a broken standing controller.
"""

env = G1Env(render_mode="human")
obs, _ = env.reset()

for _ in range(5000):
    # random_action = np.random.uniform(-1,1,29)
    env.step(np.zeros(29))
    # sleep(1)
    print(float(env.data.qpos[2]))
    env.render()
env.close()