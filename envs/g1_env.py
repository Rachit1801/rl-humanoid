import os
import numpy as np

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

MODEL_PATH = os.path.join(os.path.dirname(__file__),"..","assets","scene_29dof.xml")

TORQUE_LIMITS = np.array([
    88, 88, 88, 139, 50, 50,        # Left Leg
    88, 88, 88, 139, 50, 50,        # Right Leg
    88, 50, 50,                     # Waist
    25, 25, 25, 25, 25, 5, 5,       # Left Arm
    25, 25, 25, 25, 25, 5, 5,       # Right Arm
], dtype=np.float64)                # shape (29,)

STANDING_HEIGHT = 0.793             # from XML:  pos="0 0 0.793"

STANDING_POSE = np.zeros(29)
STANDING_POSE[0]  = -0.2    # left  hip pitch
STANDING_POSE[3]  =  0.4    # left  knee
STANDING_POSE[4]  = -0.2    # left  ankle pitch
STANDING_POSE[6]  = -0.2    # right hip pitch
STANDING_POSE[9]  =  0.4    # right knee
STANDING_POSE[10] = -0.2    # right ankle pitch

class G1Env(MujocoEnv):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):

        observation_space = Box(low=-np.inf, high=np.inf, shape=(69,), dtype=np.float64)

        super().__init__(model_path=MODEL_PATH, frame_skip=5, observation_space=observation_space, render_mode=render_mode)

        self.action_space = Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)

        self._smoothed_action = np.zeros(29, dtype=np.float64)      # EMA smoother

    def _get_obs(self):

        return np.concatenate([self.data.qpos[2:], self.data.qvel])     # Skip global x, y as we are learning to stand

    def reset_model(self):

        qpos = np.zeros(self.model.nq)
        qpos[2] = STANDING_HEIGHT # z
        qpos[3] = 1.0             # quaternion w
        qpos[7:] = STANDING_POSE.copy() # self.np_random.uniform(-0.05, 0.05, size=29)

        qvel = np.zeros(self.model.nv) # self.np_random.uniform(-0.05, 0.05, size=self.model.nv)

        self.set_state(qpos, qvel)
        self._smoothed_action[:] = 0.0 

        return self._get_obs()

    def step(self, action):

        self._smoothed_action = 0.8 * self._smoothed_action + 0.2 * action

        target_q = STANDING_POSE + 0.1 * self._smoothed_action

        q = self.data.qpos[7:]      # joint positions
        qd = self.data.qvel[6:]     # joint velocities
        kp = 50.0
        kd = 10.0
        torque = kp * (target_q - q) - kd * qd
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.do_simulation(torque, self.frame_skip)

        obs = self._get_obs()
        
        height = self.data.qpos[2]
        height_reward = float(np.clip(height/STANDING_HEIGHT, 0.0, 1.0)) * 2
        xmat_zz  = float(self.data.body("pelvis").xmat[8])   # 1.0 = upright, 0.0 = 90°
        upright = xmat_zz * 2.0
        energy = 0.001 * float(np.sum(self._smoothed_action ** 2))
        vel_penalty = 0.005 * float(np.sum(qd ** 2))

        reward = 0.5 + height_reward + upright - energy - vel_penalty

        terminated = bool(height < 0.35 or xmat_zz < 0.5)
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)

def make_env(rank: int):
    def _init():
        env = G1Env(render_mode=None)
        env.reset(seed=1000 + rank)
        return env
    return _init