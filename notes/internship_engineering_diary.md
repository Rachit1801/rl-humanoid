# DRDO Humanoid Reinforcement Learning Internship: Reconstructed Engineering Diary

## Document purpose

This document reconstructs the engineering journey recorded in the `rl-humanoid` repository and in the separate personal account at `C:\Users\admin\Desktop\MarkDown\story.txt`. It is intentionally not a repository summary. It is a chronological technical history: what was learned, what was attempted, what failed, why the design changed, what the training artifacts show, where memory and Git disagree, and which facts still require clarification.

The companion file `notes/commit_diff_ledger.md` is the exhaustive mechanical audit. It contains all 56 main-repository commits, every changed-path status, per-file text line counts, commit bodies, author and committer dates, and the internship-authored commits inside the `unitree_rl_mjlab` submodule.

## Project identity

- Organization: Defence Research and Development Organisation (DRDO).
- Guide named by the story: Dr. Prakash Kumar Uttam.
- Repository wording: “Prakash K Uttam, DRDO.”
- Domain: reinforcement learning for humanoid robots.
- Main robot: Unitree G1, 29 actuated degrees of freedom.
- Original software stack: Python, MuJoCo, Gymnasium, Stable-Baselines3 PPO, PyTorch, TensorBoard.
- Later software stack: the `unitree_rl_mjlab` fork, built on `mjlab`, MuJoCo/MuJoCo Warp, PyTorch and RSL-RL, with an Isaac Lab-inspired manager API.
- Physical scenario: balance and locomotion on a translating platform intended to approximate the floor acceleration of a moving bus.
- Repository period: 21 May 2026 through 15 July 2026. Some commit subjects refer to work done before the first Git commit, such as “19th May 2026.”

## Evidence and reliability

The reconstruction uses four evidence levels.

1. **Primary repository evidence**: commit objects, parent relationships, patches, tracked source, model ZIP metadata, TensorBoard events, monitor CSV files, videos, and submodule pointers.
2. **Contemporaneous notes**: especially `notes/log.txt`, whose entries include dates and the developer’s immediate interpretation of failures.
3. **Retrospective personal account**: `story.txt`. It supplies intent and experience that code cannot reveal, but its ordering and technical wording are not assumed to be exact.
4. **Later generated narrative**: `notes/github commits.md`. It is useful only as a list of topics. Its hashes, dates, ordering, and several conclusions conflict with Git and are not treated as authoritative.

Terminology used below:

- **Confirmed** means directly present in a patch, model, event file, or contemporaneous log.
- **Supported** means the repository strongly implies the claim but does not prove the developer’s motive.
- **Story-only** means the personal account is the only available evidence.
- **Contradicted** means primary repository evidence points the other way.

## Repository-wide audit result

- Main history: 56 reachable commits, one linear branch, no tags, no unreachable commits found.
- Commit distribution: 19 commits in May, 32 in June, and 5 in July 2026.
- Historical change surface: 423 unique paths.
- Textual churn: 32,755 inserted and 25,794 deleted lines.
- Binary diff records: 533, largely STL meshes, PPO ZIP files, VecNormalize pickles, TensorBoard events, videos, PDFs, images, and committed Python bytecode.
- Historical experiment inventory: 49 unique TensorBoard event blobs and 105 unique Stable-Baselines3 model ZIP blobs under `models`.
- Change-status records: 394 additions, 147 modifications, 195 deletions, 22 renames, and 6 detected copies.
- Current tracked tree: 205 files, including 66 under `assets`, 56 under `models`, 28 under `tb_logs`, 17 Git-tracked entries under `notes`, 11 under `envs`, 10 videos, and one submodule gitlink.
- The main repository has no `.gitignore`. This explains the repeated tracking of `__pycache__`, checkpoints, event files, and other generated artifacts. Those artifacts are poor repository hygiene, but they preserve unusually rich experimental evidence.
- Eight historical monitor CSVs preserve 16,713 episodes with a mean length of only 60.76 steps. This is direct evidence of the repeated-fall regime before the standing reset fix.

## The complete engineering timeline

### Phase 1 — Research orientation and problem definition

#### 21 May 2026

- `9aa52dd6` — **Initial commit**. Created the repository with a minimal README.
- `2f8ed303` — **19th May 2026**. Added `Introduction.md`, beginning the written survey of humanoid anatomy, history, training methods, actuation, ZMP and locomotion.
- `a7872333` — **Update README.md**. Defined the project as reinforcement-learning research for a humanoid robot under Prakash K. Uttam at DRDO.
- `c0c6d8dd` — **Updated with Latest Information**. Expanded the humanoid survey with contemporary robots and references.

#### 22–24 May 2026

- `5aaf8d1a` — **Updated Repo**. Added material about training and relevant repositories.
- `8d01f24d` — **Removed non Bipate bots**. Narrowed the survey by removing non-biped examples. The subject contains the original spelling, but the engineering decision is a scope reduction toward humanoids.
- `bbb6efbc` — **22 May 2026**. Added `rl-algorithms.md`, documenting MPC, WBC, imitation learning and the distinction between classical and learned control.
- `c6985dc8` — **Added Open Source Projects**. Added candidate open-source humanoid projects and MuJoCo references.
- `1a70815e` — **Improved Content**. Pruned and refined the project survey.

#### 25–27 May 2026

- `211ac018` — **RL Hand Written Notes**. Added `rl_notes.pdf`, explicitly sourced from a learning playlist and written in CollaNote. The notes cover backpropagation, policy gradients, PPO, cross-entropy methods and RLHF.
- `f1066b12` — **Removed HRP**. Removed HRP material after recognizing that the cited walking controller relied on MPC, LIPM and quadratic programming rather than reinforcement learning. This is an early example of rejecting a technically interesting but out-of-scope approach.
- `6f393da3` — **Add reinforcement learning workflow**. Added the conceptual pipeline: policy training, simulation, transfer to hardware, and hardware testing.
- `0b54c9f1` — **26 May 2026**. Added CartPole and Gymnasium/PPO notes.
- `e44522ed` — **MuJoCo Basic**. Added the first executable simulation (`box.py`), the MuJoCo notebook, and Python dependencies. The box-fall experiment established the core MuJoCo model/data/step/viewer loop.

**Interpretation:** The first week was not aimless background reading. It progressively narrowed the project from general humanoid robotics to RL-based control, established the simulator-interface-algorithm stack, and rejected classical HRP material because it did not match the assigned domain.

### Phase 2 — MuJoCo and Gymnasium through progressively harder toy systems

#### 28–30 May 2026

- `c34224d2` — **cartpole**. Added a MuJoCo CartPole script and expanded the notes.
- `86bc6157` — **Actuator**. Added actuator definitions and Python control of `data.ctrl`.
- `633610e7` — **Working Cartpole**. Reached a manually controllable CartPole.
- `b285b022` — **Added Pendulum**. Added `pendulum.xml` and `model_tester.py`.
- `f4f1dd14` — **Fixed Code**. Added timing/sleep behavior and stabilized the viewer loop.

The personal story’s invisible-wall diagnosis is technically plausible and consistent with MuJoCo plane semantics: a plane geom’s collision surface is infinite even if its rendered size appears bounded. Git confirms the pendulum asset and tester, but the exact stuck-and-shot-up failure exists only in the story and later narrative, not in a contemporaneous source comment.

#### 3–6 June 2026

- `8fb61d24` — **Gymnasium**. Added extensive Gymnasium learning material and a 2,803-line robotics guide explicitly attributed in the file to Claude. This is relevant to authorship and source-provenance reporting.
- `afbc2961` — **Gym for CartPole**. Added the first custom `MujocoEnv`: observation extraction, reset, action application, reward and termination.
- `b6532dcc` — **PPO implemented**. Added a trained Stable-Baselines3 PPO model. The embedded model metadata records 100,352 timesteps, one environment, `n_steps=2048`, `batch_size=64`, a four-dimensional observation, and a one-dimensional action.
- `59bfe521` — **double cartpole**. Copied and extended the XML into a double inverted-pendulum cart, expanded observations from four to six values, retrained a 100,352-step model, and added a demonstration video. The first reward/termination rewrite still considered only the first pole, so the second pole was observed but not yet part of the objective.
- `7d1b48ae` — **double pole balance**. Added a shaped two-pole objective, reduced actuator gear, trained a 501,760-step model, added a jerk test, and saved both uncontrolled-chaos and successful-balance videos. The reward source marks AI assistance, which should be acknowledged in code provenance.
- `a6078687` — **Organized files**. Moved research notes and videos into dedicated directories.
- `5b31ac76` — **Parallel Training**. Added `SubprocVecEnv` experimentation, eight seeded Windows worker processes, TensorBoard output, and a fresh 503,808-step model with `n_steps=512`. This was CPU process parallelism; there is no CUDA simulator/device configuration in this phase.
- `e04f97db` — **parallel training and organized files**. Created the durable `assets`, `envs`, `models`, `notes`, `train.py`, `run.py`, and `run_in_mujoco.py` structure; moved the toy XML files; reverted the active modular RL example from six-dimensional double-pole to four-dimensional single-pole; saved a 65,536-step model; and imported the G1 MJCF. The G1 file referenced meshes that were not added until the next commit, so this intermediate tree could not fully load the robot.

**Result:** The toy sequence deliberately increased systems complexity: rigid body, actuated cart-pole, custom RL environment, double pendulum, then parallel PPO. It also taught an important separation of concerns:

```text
MuJoCo physics model
  -> Gymnasium reset/step/observation/reward contract
  -> Stable-Baselines3 PPO
  -> vectorization, normalization, checkpoints and TensorBoard
```

The toy phase also contains technical debt that the story omits: hard-coded paths, a literal `{step}` logging string, a `render_mode="None"` typo, an undefined step variable in one script, tracked CPython caches, and a UTF-16 `requirements.txt` that Git therefore reports as binary.

### Phase 3 — Importing and understanding the Unitree G1

#### 7 June 2026

- `41a91190` — **Unitree G1 29 dof**. Added the full STL mesh set, `scene_29dof.xml`, and updated `model_tester.py` to visualize the robot. A richer obstacle/heightfield scene was also present in this transition but was malformed and referenced missing heightfield images; the flat scene was the usable path.

The G1 introduced:

- A floating base: three translations, a quaternion, and six base velocity coordinates.
- 29 actuated hinge joints: 12 legs, 3 waist, and 14 arms/wrists.
- Joint-dependent effort limits: approximately 139 Nm at knees, 88 Nm at major hip/waist joints, 50 Nm at ankles and smaller torso axes, 25 Nm in major arm joints, and 5 Nm at small wrist axes.
- A 36-element `qpos` and 35-element `qvel` before platform joints were added.

#### 8 June 2026 — Direct torque control fails

- `4cb3e039` — **G1 Standing**. Added the first G1 Gymnasium environment, trained model, `MUJOCO_LOG.TXT`, `notes/Unitree_g1.md`, `notes/log.txt`, and switched `train.py`/`run.py` from CartPole to G1.

The first control architecture let PPO actions directly determine motor torques. Early random policy outputs could simultaneously command large opposing joint torques. MuJoCo recorded:

```text
Nan, Inf or huge value in QACC at DOF 0
```

The contemporaneous log records the decision at 21:24 on 8 June: “I have tried many variations but nothing works, I will implement PD controls now.”

The first environment returned 69 observations (`qpos[2:]` plus all `qvel`) and 29 actions. Its model archive records 1,015,808 steps. The warning log contains 31 QACC instability records. Its termination expression also used an already weighted upright term, making the effective orientation threshold looser than the accompanying comment. This allowed some badly tilted states to continue longer than intended.

**Root cause:** Direct torque is a poor exploration interface for a high-DOF humanoid. PPO initially samples uncoordinated actions. Scaling those samples to hardware-scale effort limits injects extreme energy into a contact-rich, tightly coupled mechanism. The instability was a control-interface failure, not simply a PPO hyperparameter problem.

### Phase 4 — PD control, vibration suppression and the standing breakthrough

#### 9 June 2026

- `5bf3dcc9` — **PD implemented**. Changed policy actions from raw torques to desired joint-position offsets, computed clipped PD torques, added `humanoid_tester.py`, retrained, and updated the log.
- `43cb5e2c` — **smoothed_action**. Added an exponential moving-average action filter, stronger damping/reward shaping, TensorBoard runs and further training.
- `3bfc4766` — **it falls**. Removed the MuJoCo instability log because explosions were solved, but recorded that the higher-level balance objective still failed.

The new controller was:

```text
target joint position = standing pose + action scale * PPO action
torque = kp * (target - measured position) - kd * measured velocity
torque = clip(torque, hardware effort limits)
```

This decision separated high-level learned balance from low-level actuator stabilization. `humanoid_tester.py` exercised the controller without PPO. The story’s claim that it held approximately 1,200 simulation steps before toppling is story-supported and consistent with the tester’s purpose. The key distinction was learned correctly: a PD controller can hold a pose, but cannot infer center-of-mass corrections or select recovery steps.

The tester’s committed loop sampled new random actions rather than applying the zero-action hold suggested by its comment. Therefore “the PD controller alone held for 1,200 steps” needs clarification: the archived tester was a random-target robustness test, not a pure fixed-pose test.

The log captures the remaining problem:

```text
PD control finally works and now the robot is not breaking its bones
Its vibrating like someone pointed a gun at him
Tried smoothed_action... Changed reward... Doesn't stand
```

The EMA filter, higher derivative gains, energy penalties, joint-velocity penalties and action penalties attacked high-frequency oscillation. They removed one failure mode but did not solve initial-state geometry.

#### 12 June 2026 — Structured standing experiments

- `71b32754` — **g1 standing**. Centralized physical and reward constants in `g1_config.py`, added a dedicated `g1_stand_env.py`, added dedicated train/run scripts, and committed two dense checkpoint series plus VecNormalize states and TensorBoard/monitor output.
- `2e34257e` — **g1 standing**. Eight minutes later, deleted the large checkpoint collections and their normalization snapshots. This cleanup removed more than 100 generated files without changing the control code.

The two checkpoint families preserve evidence of separate training configurations:

- Ten checkpoints at 100k intervals through 1M used `batch_size=2048`, `n_epochs=5`, and no entropy coefficient.
- Sixty checkpoints at 50k intervals through 3M used `batch_size=512`, `n_epochs=10`, `ent_coef=0.005`, and a `[256, 256]` network.

This is a substantial missing experiment in the personal story. It shows that training infrastructure and hyperparameters were revised before the successful standing run.

It is not a clean reproducible snapshot. Static inspection found incompatible artifacts in the same commit: 67-observation/29-action checkpoints, 94-observation/29-action legacy checkpoints, and an 81-observation/12-action `g1_stand.zip`. The short primary config/environment also contains missing-import and state-shape/broadcast defects, while the larger alternate standing environment is internally consistent. The successful artifacts were therefore produced by one of several coexisting pipelines, not necessarily by running the shortest primary file exactly as committed.

#### 14 June 2026 — “IT WORKS”

- `0d050afe` — **it works**. Changed the standing configuration, simplified duplicate train/run files, added `g1_stand_retry.zip` and normalization state, and committed four PPO event runs.

The critical Git fact is subtle and contradicts the later narrative. The patch changed `STANDING_POSE` from an explicit bent pose to `np.zeros(29)`. The log, written two minutes before the commit, says:

```text
IT WORKS !!!
Standing pos was above so it was falling every time
```

The successful retry model records 3,014,656 timesteps. Its TensorBoard run improved mean episode length from approximately 93 to 1,646 steps and mean episode reward from about -539 to 6,296. This is strong quantitative evidence of a real standing breakthrough.

**Confirmed by the developer:** Standing was fixed by changing bent → zero so the joints matched the 0.793 m height. For walking, a bent pose is preferable, so height was lowered 0.793 → 0.78 m to compensate. Both fixes are the same underlying rule—joint pose and root height must be consistent—not opposite philosophies about bent versus straight legs.

#### 15 June 2026

- `c532b539` — **notes**. Removed the redundant `g1_stand_env.py` and documented the consolidated standing pipeline, observation vector, VecMonitor, VecNormalize and checkpoints.
- `6fbcb358` — **License**. Added an MIT license.

The final 67-dimensional standing observation was body-frame angular velocity (3), body-frame linear velocity (3), projected gravity (3), joint positions relative to the reference pose (29), and joint velocities (29). This removed absolute world x/y position and expressed motion in a policy-friendly local frame.

### Phase 5 — Push recovery as an intermediate robustness task

#### 16 June 2026

- `3fe550f7` — **Pushes**. Added randomized pelvis forces directly to the main G1 environment, committed force-trained models, two staged TensorBoard runs and `small_pushes.mp4`.
- `0950c448` — **Push Balance Working**. Refined the push logic/reward and trained another successful model.
- `d63314cc` — **g1 40N push**. Changed force settings and continued training into a second model/normalization pair.
- `abe5c34d` — **g1 120N**. Raised disturbance severity, continued into a third model, and updated evaluation.

The push implementation used `data.xfrc_applied` on the pelvis, random horizontal angle, random force magnitude, a short push duration, randomized intervals and a grace period. During the push and recovery window, posture, COM and angular-velocity penalties were relaxed so recovery motion was not punished as if it were ordinary standing.

Commit `0950c448` also corrected a training-continuation mistake: the prior path could instantiate a fresh policy instead of loading the intended standing policy and normalization state. This matters because push recovery was supposed to fine-tune balance, not relearn standing from scratch.

The historical event files show staged difficulty:

- The 15–30 N band reached a final mean episode length of approximately 1,854.
- The 30–60 N band reached approximately 1,722.
- The 60–120 N band reached only approximately 689, a material robustness regression rather than an unqualified success.
- Separate stage1/stage2 event runs also show harder disturbances reducing final mean length and reward.

The story calls this curriculum learning. The repository supports a **manual staged curriculum** through sequential models and changing force constants. The checked-in push environment does not contain an automatic curriculum scheduler. That distinction matters.

### Phase 6 — Moving platform: failed mocap design and physics-driven replacement

#### 17 June 2026 — Creating a branch point

- `5e7f10a8` — **platform xml**. Copied the existing scene into `platform_29dof.xml`; copied the push configuration/environment into `g1_config_push.py` and `g1_env_push.py` so the successful push task remained reproducible; then repurposed the main G1 environment for platform work. At this exact commit the platform was only a static world box—there was no mocap flag, slide joint, actuator or new training artifact yet.
- `4a11960c` — **platfrom working**. Implemented a moving mocap platform and documented the frame-skip/velocity issue in MuJoCo notes and the log.

This commit is when the repository first records the insight that mocap position updates do not produce ordinary dynamic joint velocity. The later `notes/github commits.md` calls this note “prophetic,” but it was written during the same commit as the experiment; it documents the diagnosis rather than predicting it.

The log states:

```text
had to write custom simulation function because frame skip was 5
velocity zero for 5 frames
I couldn't include the platform in the action space as training is already done
```

The design constraint was backward compatibility. The standing policy expected 67 observations and 29 actions. Adding platform slide joints/actuators naïvely would expand raw state/control vectors and invalidate the policy.

#### 18 June 2026 — Mocap fails

- `023abeb5` — **Robot sliping**. Added `run_platform.py` to test the mocap approach. The typo is in the original commit message.
- `6ff2fb63` — **g1 platform**. Deleted the mocap test runner, replaced the platform with two physical slide joints and two velocity actuators, excluded those joints from policy observations and actions, concatenated two externally computed platform controls after the 29 robot torques, and trained the first platform policy.

**Why mocap slipped:** Repositioning a mocap body produces geometric displacement but not the same dynamically integrated velocity state as a slide joint. With multiple physics substeps per environment action, the contact solver lacks the continuously represented tangential surface velocity needed for realistic friction. Matching render and physics rates did not repair the missing dynamics.

**Final SB3 platform mechanism:**

- Platform has x/y slide joints.
- Platform has x/y velocity actuators.
- PPO still outputs 29 robot joint targets.
- The environment computes 29 PD torques.
- The environment independently ramps a two-dimensional platform velocity.
- The MuJoCo control vector is `[29 robot torques, 2 platform velocity targets]`.
- Robot observation slices stop before the two platform joint coordinates.
- COM drift is measured relative to the moving platform body.

The first platform model records 2,015,232 timesteps. Its event runs show one weak attempt, one run improving mean episode length from about 491 to 868, and one stronger run improving from about 1,567 to 1,771.

#### 19–22 June 2026

- `cf275ae1` — **organised files**. Moved sandbox and tester scripts into the directory layout that remains today.
- `68105416` — **SSH Key**. The only repository content change is a log entry saying the SSH key was added. This is environment/setup history, not a robot algorithm change.
- `84958cb3` — **update**. Removed tracked bytecode, continued platform training to `g1_platfrom_v_2.zip`, renamed standing notes, consolidated Gymnasium notes into `librarys.md`, added a copied official `velocity_env_cfg.py`, and removed older push event files.

The second platform model also records 2,015,232 timesteps. Its TensorBoard run improved mean episode length from approximately 528 to 1,186 and reward from approximately 2,119 to 5,067.

**Outcome:** The SB3 platform policy could remain upright under smooth platform motion, but the story says it could not take recovery steps under rough disturbances. The code supports this explanation: standing reward terms favored pose retention, low action, low velocity and low COM drift. There was no explicit alternating gait objective, foot clearance target, or stepping command in this standing policy.

### Phase 7 — Walking attempt 1: history stacking and continuous curriculum

#### 23 June 2026

- `97cd0c48` — **30M Training**. Added the first dedicated walking configuration/environment/trainer, a 639-line reverse engineering of Unitree’s Isaac Lab velocity task, an initial locomotion blueprint, two large TensorBoard runs and the “30M overnight” log entry.

This first walking architecture was significantly more ambitious than the personal story conveys:

- 99 values per frame: platform-relative body linear velocity, body angular velocity, projected gravity, 3D velocity command, 29 relative joint positions, 29 joint velocities and 29 previous actions.
- Five-frame history stack, for a declared 495-dimensional observation.
- Source comments and console text incorrectly say 480 in places; the model/config say 495.
- Eight CPU subprocess environments.
- PPO `[512, 256, 128]`, 4,096 rollout steps per environment, 512 batch, 10 epochs, entropy 0.01, 30M intended steps.
- Domain randomization of platform/foot friction and torso mass.
- Velocity-command resampling.
- Platform-relative velocity tracking and platform-relative foot slip.
- Gait phase, foot clearance, undesired-contact, joint-deviation, action-rate, energy, joint-limit, COM drift and termination terms.
- Continuous curriculum level increased only when an episode survived at least 80% of its horizon and achieved a tracking threshold.

The event data does not show a successful curriculum. `curriculum/level` stayed exactly 0 throughout the long run. Mean episode length fell from about 83 to 34 by 27.8M logged steps. Policy standard deviation rose from approximately 1.0 to 789,402,176. The contemporaneous log summarizes the result as “30M steps overnight just for the mean_episode length of 50.”

This is stronger evidence than a simple “insufficient training” explanation. The optimization was unstable and never unlocked harder curriculum.

### Phase 8 — Walking attempt 2: import official code, then re-adapt it

#### 24 June 2026

- `2e43d5a6` — **official code**. Deleted the first custom walking implementation and copied a snapshot of official reference files into `official code/`: environment configuration, observations, rewards, curriculum, commands, terminations, G1 constants/XML, train/play and PPO configuration. It also archived a walking model, normalization state and three event runs.

The archived `g1_walk_v1.zip` from this phase has a much larger observation space than the later 103/115-dimensional SB3 models and a low initial action standard deviation. It is an artifact of a distinct architecture, not merely the same code under a new filename.

#### 28 June 2026

- `889fb64c` — **Progress**. Reintroduced a custom Gymnasium/SB3 walking implementation after studying the copied reference. Added models v1/v2, normalization states, evaluation, a screenshot and eleven event logs.

The new implementation reduced observation complexity to 103 dimensions and introduced:

- A bent-knee 0.78 m reference pose.
- PD gains computed from reflected inertia rather than copied high gains.
- Phase clock and command resampling.
- Contact-pair-based foot detection.
- Adaptive staged velocity/platform curriculum.
- A custom asymmetric policy that hid platform velocity from the actor.
- PPO `[512, 256, 128]` with ELU, 2,048 rollout, 512 batch, 5 epochs and entropy 0.01.
- An alive reward of 1.0.

Historical model metadata reveals an experiment not preserved in current source: `g1_walk_v2.zip` used a Gymnasium `Dict` observation with separate actor and critic spaces and a custom asymmetric dictionary policy. The committed source around it uses a flat observation. This likely represents an uncommitted intermediate design.

The many runs show both apparent improvement and pathological exploration:

- One 10M run improved episode length from about 110 to 724 and reward from -768 to 2,015, while policy standard deviation grew to 7.38.
- One 20M run collapsed from about 101 to 34 mean steps while standard deviation grew to 164,959.
- Other 10M runs produced enormous approximate KL values, including tens of millions or billions, even while standard deviation remained near 1. These are signs of numerically destructive PPO updates.

### Phase 9 — Walking attempt 3: survival-reward exploitation and the external pivot

#### 2 July 2026

- `26029f9f` — **external repo**. Added the `unitree_rl_mjlab` submodule at `a1cca58`, removed the copied `official code` directory, added Conda notes, revised custom walking code, saved v3/v4 policies and event logs, and added resume/version support.
- `d9ed5195` — **video**. Added `g1_drunk walk.mp4` and `g1_walk.mp4`.

The reward revision reduced the alive bonus from 1.0 to 0.25, replaced upright reward with orientation penalties, added yaw tracking, joint acceleration, joint limits, foot clearance and soft landing, clipped reward, and strengthened the official-style structure.

The v3/v4 sequence quantitatively confirms the “drunk robot” story:

- The first 20M segment finished near 1,084 mean episode steps and policy std 282.
- The resumed next 20M segment began around std 285 and ended around 55,043.
- The resumed third 20M segment began around std 55,778 and ended around 11,772,209.
- Episode length remained around 1,100 and reward remained positive, so the policy was learning to survive while its action distribution became wildly random.

Because `model.learn()` reset the per-run timestep counter on resume, each saved ZIP reports about 20M steps even though v1/v3/v4 represent an approximately 60M cumulative sequence. This is why the personal story’s 60M description and the individual ZIP metadata initially look inconsistent.

The repository also contains an earlier independent run that reached std 789M by 27.8M steps. Therefore the exact phrase “reached millions after 60M” applies well to the v3/v4 sequence, but was not the only standard-deviation explosion.

### Phase 10 — Final custom SB3 rewrite and final `unitree_rl_mjlab` integration

#### 8 July 2026

- `84c54242` — **Custom Walk**. Performed a final large SB3 rewrite, added `notes/Repository Overview.md`, and committed v5 checkpoints every 100k through 1.5M.
- `4ab9f3fa` — **Custom Walk**. Updated the submodule pointer from `a1cca58` to merge commit `5d234423`.

The final SB3 rewrite:

- Removed alive, height, energy, joint-velocity and raw action-magnitude rewards.
- Increased the fall penalty from -50 to -200.
- Changed adaptive survival-based curriculum to a two-stage step-based curriculum.
- Changed observation from flat 103D to 115D: 98 actor features and 17 critic-only features.
- Added critic-only body linear velocity, platform velocity, foot height, air time, contacts and contact forces.
- Used separate actor/critic feature extractors and masked critic-only input for the actor.
- Matched official reward weights more literally.

However, in matching the flat-ground official task, it removed platform-relative velocity tracking and platform-relative foot-slip calculation. The final v5 run stopped at about 1.49M logged steps; mean episode length fell from about 24 to 18 and reward stayed strongly negative. No final `g1_walk_v5.zip` exists—only checkpoints through 1.5M. This attempt was unsuccessful.

The checked-in `train_walk.py` docstring still claims per-environment promotion/demotion, while the implementation is step-based and only promotes once. Because the counter belongs to each of eight subprocess environments, the 120k threshold corresponds to roughly 960k aggregate SB3 timesteps. `run_walk.py` says to select stage 4 even though only stages 0 and 1 exist, and it still loads a 103D v4 model/normalizer after the live environment changed to 115D. The runner is therefore not compatible with the final environment snapshot.

Two more implementation details limit how the final custom results should be interpreted. The six privileged “contact force” values reconstruct only each contact’s normal component, not full normal-plus-friction XYZ forces. Checkpoint callback frequency is 12,500 vector calls across eight workers—100,000 aggregate timesteps, matching the filenames—not one checkpoint per 12,500 environment steps.

#### Submodule work on 8 July 2026

The submodule contains four internship-authored commits after its external upstream base:

- `a1cca58b` — **my_env**, 2 July. Imported a temporary `My Code` directory containing the custom G1 environment, platform MJCF, mesh copy, ONNX/Viser runner and configuration, plus a large quantity of Python bytecode.
- `54c5e8c1` — **Walking**, 8 July 15:53. Deleted the temporary `My Code` tree and integrated platform support into the manager-based codebase.
- `147059e6` — **Walking**, 8 July 16:10. Repeated the same tree change on a parallel branch.
- `5d234423` — merge, 8 July 16:10. Merged the two duplicate “Walking” commits. The duplication likely came from committing equivalent local and remote work before pulling/merging, but the motive needs confirmation.

The integrated external design added:

- `platform_g1.xml`: physical x/y slide joints and velocity actuators.
- A registered `Unitree-G1-Platform` task.
- Collision and friction configuration for feet/platform.
- Filtering that prevents the two platform joints from entering robot actions, observations, joint resets, posture reward, joint-limit reward or acceleration reward.
- Smooth randomized platform target velocity with hold periods and controllable ramp rate.
- Platform-relative command tracking: subtract platform world velocity, rotate into body frame, and compare against the commanded relative velocity.
- Platform-relative foot slip: subtract platform velocity from contacting-foot velocity.
- A curriculum that expands platform speed and acceleration sharpness up to ±2 m/s and a ramp rate of 20 m/s²-equivalent command change.
- A Viser viewer subclass with x/y velocity sliders from -2 to +2 m/s and a zero-speed button.

This implementation repaired the conceptual regression in the last SB3 version: moving-platform rewards must be expressed in the surface frame, not blindly copied from a stationary-ground task.

The external framework’s G1 PPO configuration uses separate feed-forward actor and critic networks `(512, 256, 128)`, ELU activation, observation normalization, scalar Gaussian std initialized to 1, adaptive learning rate around 1e-3, PPO clip 0.2, entropy 0.01, 5 learning epochs, 4 minibatches, 24 steps per environment and up to 10,001 iterations. The README’s training example uses 4,096 parallel environments, which explains the order-of-magnitude speedup over eight CPU subprocesses.

#### 15 July 2026

- `e39e9cb5` — **End**. Added final result videos (`Walk.mp4`, `balance.mp4`, `top view.mp4`) and `notes/github commits.md`.

The story says the final policy walked and balanced under bidirectional platform motion and that a harsher policy spread its arms under extreme motion. Videos support that result qualitatively, but no final RSL-RL checkpoint, TensorBoard/RSL log directory, exact command, seed, GPU, environment count or extreme-acceleration configuration is tracked. The claim is therefore supported by story/video naming but not reproducible from the repository alone.

## Quantitative experiment record

### CartPole and double pendulum

- The story says CartPole trained for 2,000 steps.
- The first committed PPO model records 100,352 training steps.
- The double-pendulum replacement also records 100,352 steps.
- A later parallel CartPole artifact records 65,536 steps.
- “2,000” may refer to an evaluation loop or remembered episode horizon, not model training. Clarification is needed.

### Standing

- Early direct-torque and PD runs frequently lack episode statistics because VecMonitor was not always in the wrapper chain.
- First monitored standing run: about 2.0M steps, mean length 89 -> 240, reward 41 -> 939.
- Larger standing run: about 3.0M steps, mean length 92 -> 1,518, reward -536 -> 5,407.
- Successful retry: about 3.0M steps, mean length 93 -> 1,646, reward -539 -> 6,296.
- Explained variance in successful runs approached 0.99, indicating the critic eventually modeled returns well.

### Push recovery

- Gentler stage: approximately 2.0M steps, mean length remained around 1,400–1,800.
- Continued harder stage: another approximately 2.0M steps, with reduced final survival/reward.
- Strong 120N-era event run: approximately 2.0M steps, mean length around 553 -> 689.
- Model names and sequential continuation prove staged training; no automatic push scheduler is checked in.

### SB3 moving platform

- Platform v1/v2 ZIPs each record approximately 2.015M steps.
- Best recorded platform run reached mean episode length around 1,771.
- A separate platform run reached around 1,186.
- Some runs exhibited very large PPO KL spikes, so learning was not uniformly stable.

### Walking

- Initial history-stack run: 27.8M logged steps, curriculum stayed 0, mean length collapsed, std reached 789M.
- Later custom run: 10M, mean length 110 -> 724, but std rose to 7.38.
- Another custom run: 20M, mean length 101 -> 34, std reached 164,959.
- The survival-exploitation v1/v3/v4 chain: approximately 60M cumulative, mean lengths around 1,000+, std 1 -> 282 -> 55k -> 11.7M.
- Final 115D v5 attempt: only 1.49M, mean length 24 -> 18, no final model.
- External 120M/5-hour training: developer clarification — that figure refers to a different submodule/training setup, not the SB3 history audited here; do not treat it as a claim about this repository’s TensorBoard runs.

## Repository evolution

### Documentation-first repository

The project began as a research notebook. `Introduction.md`, `rl-algorithms.md`, handwritten notes and MuJoCo/Gymnasium guides preceded most executable control code. This explains why later engineering decisions often cite specific external implementations instead of appearing as unexplained constants.

### Sandbox-to-application transition

The top-level scripts initially mixed experiments, assets and notes. Commits `a6078687`, `e04f97db` and `cf275ae1` progressively separated:

- `sandbox`: toy physics and parallel-training experiments.
- `tools`: model and controller validation.
- `assets`: MJCF and meshes.
- `envs`: Gymnasium tasks and configuration.
- `models`/`tb_logs`: learned artifacts and evidence.
- `notes`: theory, logs, reverse engineering and code explanations.

### Standing code consolidation

Standing briefly had parallel implementations (`g1_env.py` and `g1_stand_env.py`) and dedicated `g1_train.py`/`g1_run.py`. After success, those duplicates were removed and shared `train.py`/`run.py` became the maintained entry points.

### Preserving successful baselines before riskier experiments

When platform work began, the push implementation was copied into `g1_config_push.py` and `g1_env_push.py`. This preserved a working recovery baseline while the primary G1 files changed model topology. It is a sound experimental branching pattern even though Git branches were not used.

### Artifact-heavy version control

The repository repeatedly commits and deletes checkpoint sets, normalization pickles, TensorBoard events, compiled Python and large media. This made cloning and review expensive, but now permits model-configuration and metric archaeology. For a production research repository, generated artifacts should move to release/object storage and experiment tracking, with manifests and hashes committed instead.

### Pivot from monolithic environment to manager-based framework

The SB3 environment manually handled contacts, command sampling, curriculum, observations, rewards and platform actuation in one class. The external fork decomposed these into scene entities and observation, event, reward, curriculum and action managers. The pivot was architectural, not just “using a faster trainer.”

## Important implementation decisions and their engineering rationale

### MuJoCo as the simulation base

MuJoCo offered editable MJCF, contact dynamics, an accessible Python API and a path from CPU prototyping to GPU-accelerated MuJoCo Warp. The project remained in one physics ecosystem even when the RL framework changed.

### Gymnasium as the boundary

The custom environment made reset, observation, reward, termination and action semantics explicit. This allowed CartPole techniques to transfer to G1 while keeping PPO independent of MuJoCo internals.

### Position-target actions plus PD

This was the highest-impact decision. It bounded policy authority, matched common legged-robot control stacks, prevented random exploration from directly applying destructive torques, and created a plausible later sim-to-real interface.

### Body-frame and relative observations

Rotating velocities/gravity into the pelvis frame removed dependence on global heading. Subtracting the standing pose made joint state centered around zero. Platform-relative velocity was necessary because “standing still on a bus floor” means matching the floor velocity in world coordinates.

### Keep platform controls outside the policy action

This preserved the 29D action contract and treated platform motion as an exogenous disturbance. The same principle was retained in the manager-based implementation through joint/actuator filtering.

### Vectorization and normalization

Eight `SubprocVecEnv` workers made CPU training practical. `VecNormalize` stabilized differently scaled observations and rewards, but required saving/loading matching statistics. Several model/norm pairs in Git show that this dependency was understood.

### Progressive robustness tasks

The sequence standing -> external pushes -> moving platform -> walking is a rational curriculum at the project level. It isolated failures: first stabilize actuation, then learn balance, then recovery, then sustained surface motion, then stepping.

### Asymmetric actor-critic

The final custom design gave privileged simulator signals to the critic while withholding them from the actor. This helps value estimation without requiring deployment-time access to contact forces or global/base velocity. The external manager framework naturally maintains separate actor and critic observation groups.

### Abandoning the custom SB3 walking stack

The decision was justified by evidence, not impatience: repeated 20–30M-step runs, curriculum stagnation, catastrophic KL spikes, standard deviation explosions, CPU throughput limits and reward exploitation. The external framework provided thousands of GPU-parallel environments, RSL-RL conventions, tested reward organization and deployment tooling.

## Contradictions and corrections

### 1. Standing pose direction

**Story:** The fix was changing a straight all-zero pose to Unitree’s bent pose.

**Git:** The 14 June success commit changed the configured pose from bent to all zeros. Later walking code used bent knees only after lowering the spawn height.

**Correction (confirmed):** Standing: bent → zero at 0.793 m. Walking: keep bent, lower height 0.793 → 0.78 m. Same rule—pose and height must match.

### 2. CartPole training duration

**Story:** 2,000 training steps.

**Model metadata:** 100,352 steps in the first model.

**Possible reconciliation:** 2,000 may be the evaluation loop length or an early uncommitted run.

### 3. Pendulum ordering

**Story:** Pendulum came after the complete CartPole PPO pipeline and file organization.

**Git:** `pendulum.xml` and `model_tester.py` were added on 30 May, before the custom Gymnasium environment and PPO commits on 3–4 June.

### 4. Timing of the full official walking-code study

**Story:** Deep official-code recreation is narrated as part of the pre-standing struggle.

**Git:** Early G1 code did borrow Unitree PD values before standing success, but the dedicated 639-line walking reverse engineering and copied official velocity task appeared on 23–24 June, after standing, pushes and the first platform implementation.

### 5. Automatic push curriculum

**Story:** Push force was slowly increased “using a curriculum.”

**Git:** Force levels and models were changed in sequential commits. The preserved push environment has fixed random bounds and no automatic stage manager.

**Correction:** It was a manual staged curriculum unless uncommitted code existed.

### 6. Survival reward was not literally absent

**Story:** Walking failed because the robot had no reason to live, then an alive reward was added and exploited.

**Git:** The first 23 June walking configuration already had `WEIGHT_ALIVE=0.15`; the 28 June rewrite used 1.0; 2 July reduced it to 0.25; 8 July removed it and increased termination cost.

**Correction:** “No reason to live” describes insufficient net incentive under the full reward landscape, not literal absence from every version.

### 7. Platform action smoothing

**Story:** Action smoothing was added during moving-platform training.

**Git:** EMA action smoothing is explicit in the 9 June standing history. The final SB3 platform environment does not maintain a smoothed-action state; it relies on PD control and smoothly ramped platform velocity.

### 8. “Well-defined” SB3 platform curriculum

**Story:** The first moving-platform model used a well-defined curriculum.

**Git:** The SB3 platform target is randomized/ramped but not organized into explicit stages. A true multi-stage platform speed/ramp curriculum appears later in `unitree_rl_mjlab`.

### 9. External framework identity

**Story:** The selected repository was based on Isaac Lab.

**Repository:** It is based on `mjlab`, MuJoCo/MuJoCo Warp, PyTorch and RSL-RL. Its API deliberately resembles Isaac Lab, and parts of the reward/task design derive from that ecosystem, but it is not running NVIDIA Isaac Sim/Isaac Lab physics.

### 10. “Any speed”

**Story:** The final robot balanced “at any given speed.”

**Code:** Training curriculum and viewer controls are bounded to approximately -2 to +2 m/s in x/y.

**Correction:** The defensible claim is “across the tested bidirectional speed range,” not arbitrary speed.

### 11. Final success came from the submodule, not the last SB3 code

**Confirmed by the developer:** Final walking/balance success is from the cloned `unitree_rl_mjlab` fork (manager-based mjlab/RSL-RL), not from the last custom SB3 `g1_walk` rewrite. That SB3 v5 attempt failed. Supporting write-ups of the fork changes exist outside the repo (`g1 walk report.md`, `diff_report.md`): platform XML, joint filtering, platform-relative rewards/events/curriculum, and Viser platform sliders after upstream `1425b15`.

**120M / 5 hours:** Refers to a different submodule/training setup; ignore for claims about this repository’s SB3 TensorBoard history.

**Remaining gap:** RSL-RL checkpoint/logs for the demonstrated videos are still not committed in this tree.

### 12. `notes/github commits.md` is not a factual commit history

It uses nonexistent or misassigned hashes, maps late commits to early work, calls the mocap note prophetic, places “SSH Key” at G1 study, and reverses the standing-pose patch. It also describes a nonexistent position-actuator platform, a five-stage final curriculum where the source has two, 50M final runs where the committed configuration is 20M, and EMA/reward/termination behavior absent from the final source. It should not be cited as a primary timeline.

## What can be claimed in an internship report

### Strong, repository-supported claims

- Learned the conceptual and practical layers of humanoid RL: physics, environment interface, policy optimization and low-level control.
- Implemented MuJoCo CartPole and double-pendulum systems before moving to G1.
- Built custom Gymnasium environments and PPO train/evaluation pipelines.
- Used parallel CPU environments, VecMonitor, VecNormalize, checkpoints and TensorBoard.
- Diagnosed direct-torque instability and replaced it with clipped joint-specific PD control.
- Built body-frame observations and shaped standing rewards.
- Diagnosed an initial pose/root-height inconsistency and achieved long-horizon standing.
- Trained disturbance recovery with external forces and increasing manual difficulty.
- Diagnosed mocap surface-velocity/friction failure and replaced mocap with actuated slide joints.
- Preserved the robot’s 29D policy interface while adding two exogenous platform controls.
- Implemented several walking architectures, contact sensing, gait/clearance/slip rewards, command sampling, curriculum and asymmetric actor-critic.
- Used TensorBoard metrics to identify suicide and survival/randomness local optima.
- Pivoted from eight CPU environments to a GPU-parallel manager-based MuJoCo framework.
- Added platform-relative reward terms, curriculum and Viser controls to the external framework.

### Claims that require qualification

- “120M steps in five hours”: not a claim about this repo’s SB3 runs; omit or attribute only to the separate training setup the developer identified.
- “Any speed”: replace with the tested ±2 m/s Viser/curriculum range unless another configuration is recovered.
- “Curriculum” in push/SB3 platform phases: call it manual staged training or randomized difficulty, reserving automatic curriculum for code that actually schedules stages.
- “Isaac Lab repository”: call it a cloned Isaac Lab-inspired `mjlab`/MuJoCo Warp/RSL-RL framework that was extended for the moving platform.
- Deployment: simulation validation only (confirmed).

## Final chronological project story

The internship began with a broad study of humanoid mechanics, control and learning. Classical approaches such as ZMP, MPC and WBC were studied, but the repository scope was deliberately narrowed to reinforcement learning. MuJoCo was selected as the simulator, Gymnasium as the environment contract and Stable-Baselines3 PPO as the first learning implementation.

Simple simulations converted theory into practice. A falling box established the MuJoCo loop. CartPole introduced joints, actuators, sensors and Python control. A pendulum exposed the difference between visual and collision geometry. A custom Gymnasium CartPole and PPO model established observation/action/reward/reset semantics. A double inverted pendulum and parallel environments increased task and infrastructure complexity.

The imported Unitree G1 transformed the project from a tutorial into control research. Direct PPO torque actions generated catastrophic acceleration and NaNs. Reframing actions as desired joint offsets behind a PD controller stabilized the mechanism. EMA smoothing, damping and penalties reduced vibration, but the robot continued to fall because the reference pose and root height were inconsistent. Once that initialization issue was corrected, the learned policy’s episode length and reward increased dramatically.

Robustness was then introduced in stages. Random pelvis pushes taught recovery, first gently and then at higher force. A moving platform was a more faithful bus-floor disturbance, but a mocap body did not expose physical surface velocity to the contact solver, so the feet slipped. Replacing mocap teleportation with x/y slide joints and velocity actuators restored dynamic friction. Careful slicing kept platform joints out of the 67D observation and 29D robot action, allowing previous policies and network shapes to be reused.

Standing reward shaping could not teach useful recovery stepping. Walking required commands, gait phase, contact detection, clearance, slip, action-rate and joint-acceleration terms. The first walking environment stacked five frames into 495 dimensions and used continuous curriculum/domain randomization, but curriculum never advanced and PPO standard deviation exploded. The next implementation reduced observations and added an alive reward and asymmetric information, but survival became exploitable: episode length improved while random-action standard deviation grew through 282, 55,000 and 11.7 million across approximately 60M cumulative steps. A final official-weight rewrite removed alive reward and privileged the critic, but collapsed within 1.5M steps.

At that point the engineering bottleneck was no longer conceptual understanding; it was throughput, reward calibration and framework maturity. The last custom SB3 rewrite was abandoned as the success path. The project moved to a cloned GPU-parallel MuJoCo/MuJoCo Warp framework (`unitree_rl_mjlab`) with RSL-RL and manager-based task composition. The author’s additions—documented in the external `diff_report.md` / `g1 walk report.md`—integrated a physical platform, filtered robot versus platform joints, rewrote velocity and slip rewards in the platform frame, added platform velocity curriculum, and exposed Viser platform sliders. Final videos record simulated walking and balance on the moving platform. No physical G1 deployment was attempted.

## Final assessment

The most important outcome was not a single policy file. It was the progression of engineering understanding:

1. Physics state is not the same as rendered motion.
2. Action semantics determine whether exploration is physically survivable.
3. Initial-state geometry can dominate millions of RL steps.
4. Reward increases do not necessarily mean desired behavior; episode length and policy variance must be interpreted together.
5. A moving reference surface changes the correct coordinate frame for velocity and slip rewards.
6. Curriculum must be implemented and measured, not only described.
7. Framework choice and parallel simulation throughput can be decisive research variables.
8. Reusing tested infrastructure is an engineering decision when repeated experiments show that a custom stack is consuming the remaining schedule.

The repository supports a defensible internship narrative of iterative learning, failure analysis, control redesign, experimental instrumentation and a final architecture pivot. The unresolved questions above should be answered before converting this context into a formal report, because they affect exact claims about chronology, curriculum, training scale and final performance.
