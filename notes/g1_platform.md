# Balance on a Moving Platform

The final objective is to train a Unitree G1 humanoid robot that can maintain balance under real-world disturbances without using external support. The robot should be able to stand on a platform that behaves similarly to the floor of a moving bus, train, ship, or industrial vehicle.

## Plan 

1) Stand on ground
2) Stand while being pushed
3) Stand on a moving platform
4) Stand on a moving platform with unpredictable disturbances

Every few seconds Apply random pushes to pelvis. `force_x = np.random.uniform(-20, 20)` then slowly increase the random push force

## External Push Recovery for the G1

### Mechanism

MuJoCo API call `xfrc_applied` is a `(nbody × 6)` array MuJoCo exposes. The first 3 columns of each row are a Cartesian force in Newtons applied at that body's center of mass in world frame. The last 3 are torque.

```python
self.data.xfrc_applied[body_id, :3] = force_vector
```

Write `xfrc_applied`  before `do_simulation()`

`xfrc_applied[:] = 0.0` in reset prevents force bleed between episodes

Timing state machine (countdown, remaining) controls when pushes happen

Reward relaxation during push otherwise policy goes rigid

Grace period tracking needed to know when to relax rewards

Recovery bonus during grace

Push force in observation (67 to 70 dims) helps policy react proactively, but complicates weight transfer

---

When loading existing model and training over it just load your standing model instead of creating a fresh one

```python
# wrong for push training:
model = PPO(policy="MlpPolicy", env=train_env, ...)

# Replace with:
model = PPO.load("models/g1_stand_retry", env=train_env)
```

