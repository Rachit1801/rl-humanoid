# Gymnasium

Every pyhsics simulator (MuJoCo, Issac Labs) has different code. Gymnasium standardizes everything into `obs, reward, terminated, truncated, info = env.step(action)` and all environments behave similarly.

It is class with 

- **reset()**: start new episode

- **step(action)** : apply action

- **render()** : show viewer

- **observation_space** : obs dimensions

- **action_space** : action dimensions

<img src="file:///C:/Users/admin/AppData/Roaming/marktext/images/2026-06-03-01-51-36-image.png" title="" alt="" width="323">

## Structure

```python
class MyEnv(gym.Env):

    def reset(self):
        ...

    def step(self, action):
        ...
```

## Observation

What the agent sees.

Examples for a humanoid : joint angles, velocities, torso tilt, foot contact

Example: `obs = [0.1, -0.3, 1.2, 0.0]`

## Action

What the agent outputs.

Examples : motor torques, target joint positions, wheel speeds

## Reward

A score telling the agent how good its action was.

Examples: Walking robot +1 for moving forward, -10 for falling

## Episode

One complete attempt. Episode ends then environment resets.

Example: humanoid starts standing, walks, falls

```python
obs, reward, terminated, truncated, info = env.step(action)
```

obs : new state
reward : score
terminated : natural ending
truncated : timeout ending
info : debug info

# Parallel Training

Gymnasium has `from gymnasium.vector import AsyncVectorEnv`  and `from gymnasium.vector import SyncVectorEnv`

So Now

Example:

```python
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

envs = AsyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(8)]
)

obs, _ = envs.reset()
```

`obs.shape` is `(8, 4)` because you have observations from 8 CartPoles.

So PPO is `2048 steps × 8 envs `
