# Introduction

### What is Humanoid Robot?

Humanoids are general-purpose, bipedal robots.

Humanoid robots are characterized by their [anthropomorphic design](https://en.wikipedia.org/wiki/Anthropomorphism "Anthropomorphism").

- torso

- head

- two arms

- two legs

---

### Actuator

Actuators are the motors responsible for motion in the robot.

**Types:**

- **Electric**  - Small, less powerful. Hence it is common to use multiple electric actuators for a single joint in a humanoid robot.

- **Hydraulic**  - Higher power, bulky in size. One solution to counter the size issue is [electro-hydrostatic actuators](https://en.wikipedia.org/wiki/Electro-hydraulic_actuator "Electro-hydraulic actuator")

- **Pneumatic**  - Gas Based. As they inflate, they expand along the axis, and as they deflate, they contract.

---

### Training

Reinforcement learning and imitation learning are used to train robots to perform tasks like grasping objects or navigating obstacles

**Imitation Learning:** Robots can acquire new skills by replicating movements demonstrated by humans.

**Reinforcement Learning:** An algorithm uses a mathematical equation to reward robots for correct actions and penalize them for incorrect actions through trial and error.

---

### Control

To maintain balance during the walk a robot needs information about contact force and its current and desired motion. The solution to this problem relies on a major concept, the [Zero Moment Point](https://en.wikipedia.org/wiki/Zero_Moment_Point "Zero Moment Point") (ZMP).

Humanoids should be able to move in complex environments, planning and control must focus on self-collision detection, path planning and obstacle avoidance. I have written a editorial on [A-star algorithm](https://nimav.at/courses/2026-spring-cs405/editorials/a-star/) which helps in path planning and obstacle avoidance.

---

### History

- **1961** : [Unimate](https://en.wikipedia.org/wiki/Unimate "Unimate") ; The first digitally operated and programmable non-humanoid robot.

- **1972** : WABOT-1 ; world's first full-scale humanoid intelligent robot. It was the first android able to walk, communicate with a person in Japanese (with an artificial mouth), measure distances and directions to the objects using external receptors (artificial ears and eyes), and grip and transport objects with hands.

- **1984 : WABOT-2** ; musician humanoid robot able to communicate with a person, read a musical score, and play an electronic organ.

- **1985 : WHL-11** ; biped robot capable of static walking on a flat surface and turning.

- **1986 : Honda E series** ; early experimental biped humanoid robot series by Honda.

- **1993 : Honda P series** ; advanced prototype humanoid robots with upper limbs.

- **1995 : WABIAN** ; full-scale human-like walking robot from Waseda University.

- **1997 : Hadaly-2** ; interactive humanoid robot focused on human communication.

- **2000 : ASIMO** ; iconic humanoid robot by Honda capable of walking and running.

- **2001 : QRIO** ; entertainment humanoid robot developed by Sony.

- **2001 : HOAP** ; Fujitsu’s compact humanoid research robot platform.

- **2002 : HRP-2** ; advanced biped humanoid robot for robotics research.

- **2003 : Actroid** ; highly realistic android robot with silicone skin.

- **2004 : KHR-1** ; programmable hobby humanoid robot by Kondo Kagaku.

- **2005 : HUBO** ; South Korean walking humanoid robot developed by KAIST.

- **2005 : PKD Android** ; humanoid robot modeled after author Philip K. Dick.

- **2006 : NAO** ; small programmable humanoid robot widely used in education and research.

- **2006 : iCub** ; open-source humanoid robot designed for cognition research.

- **2007 : TOPIO** ; humanoid robot capable of playing table tennis.

- **2008 : Justin** ; humanoid robot developed by the German Aerospace Center (DLR).

- **2008 : Nexi** ; social humanoid robot combining mobility, dexterity, and interaction.

- **2008 : Surena** ; Iranian humanoid robot capable of speech and object tracking.

- **2009 : HRP-4C** ; realistic Japanese humanoid robot capable of singing and dancing.

- **2009 : DARwIn-OP** ; open-source humanoid robot platform for research and education.

- **2010 : Robonaut 2** ; NASA humanoid robot designed for space missions.

- **2010 : REEM** ; autonomous humanoid service robot by PAL Robotics.

- **2011 : ASIMO (2nd Generation)** ; upgraded ASIMO with semi-autonomous capabilities.

- **2012 : NimbRo-OP** ; open humanoid robot platform for autonomous robotics research.

- **2013 : TORO** ; torque-controlled humanoid robot developed by DLR.

- **2013 : Valkyrie (R5)** ; NASA humanoid robot for future space exploration missions.

- **2014 : Manav** ; India’s first 3D-printed humanoid robot.

- **2016 : Sophia** ; AI-powered humanoid robot known for realistic conversations.

- **2016 : OceanOne** ; underwater humanoid robot with haptic feedback capabilities.

- **2017 : TALOS** ; industrial humanoid robot for advanced manipulation tasks.

- **2018 : Rashmi Robot** ; Indian multilingual humanoid robot with emotional interaction.

- **2020 : Digit** ; humanoid robot for logistics and autonomous delivery tasks.

- **2020 : Vyommitra** ; Indian humanoid robot developed for the Gaganyaan space mission.

- **2020 : Robot Shalu** ; multilingual Indian humanoid robot built using recycled materials.

- **2022 : Ameca** ; highly expressive humanoid robot by Engineered Arts.

- **2022 : Optimus** ; Tesla’s general-purpose humanoid robot initiative.

- **2024 : Atlas (Electric)** ; fully electric humanoid robot by Boston Dynamics.

- **2024 : G1** ; affordable humanoid robot by Unitree Robotics.

- **2024 : HumanPlus** ; humanoid robot capable of mimicking human movements.

- **2025 : SE01** ; humanoid robot capable of performing forward flips.

- **2025 : Figure 03** ; autonomous humanoid robot designed for household chores.

- **2025 : NEO** ; consumer-ready humanoid robot for home assistance.

- **2025 : AgiBot A2** ; humanoid robot that set a long-distance walking world record.

---

### Zero moment point

ZMP is the point on the ground where the robot’s total tipping moment becomes zero.

If the ZMP stays inside the support area (area covered by the feet touching the ground), the robot remains stable.

If the ZMP moves outside the support area, the robot starts to tip or fall.

![The ZMP is an axis, not a point](https://scaron.info/images/cmp-zmp-popovic-herr.png)

---

# Latest and Important Humanoid Robots

### Moya

Company : DroidUp (China)

World’s first fully biomimetic embodied intelligent robot.

Designed to move, react, and exist in a way that feels human physically and socially. It reproduces human micro-expressions and subtle timing.

### G1

Company : Unitree Robotics (China)

A compact humanoid (127 cm, 35 kg) that completed a 130,000-step autonomous trek in extreme cold (-47.4°C). It uses a Bato satellite navigation system and runs on Unitree's Unifol LM model.

### Themis Gen 2.5

Company : Westwood Robotics

A robot designed to manipulate objects *while* walking. It uses an AI-augmented humanoid operating system (AOS) that integrates perception, planning, and control. It features an object-centric vision action model (OC VAM) and upgraded hardware, including arms that can handle 5 kg payloads and new "Mountain Bear" actuators in the hips.

### Atlas

Company: Boston Dynamics / Hyundai (USA/South Korea)

Most advanced mobility and athletic performance in a commercial humanoid.
Fully electric with exceptional range of motion (high DoF, including 360° joint capabilities), strength, and whole-body control. Powered by AI (e.g., Google DeepMind partnership). Designed for industrial tasks

### Other

**Optimus**: Tesla (USA) ; Mass-market general-purpose humanoid.

**Figure 03** : Figure AI (USA) ; general-purpose AI humanoid for homes and industry.

**NEO** : 1X Technologies (USA/Norway) ; major consumer/home humanoid robot.

**LG CLOiD** : Strong home task demonstration (laundry, dishes) 

--- 

# Repos

- [Reinforcement Learning for Humanoid Robot](https://github.com/roboterax/humanoid-gym) : reinforcement learning (RL) framework based on Nvidia Isaac Gym, designed to train locomotion skills for humanoid robots, emphasizing zero-shot transfer from simulation to the real-world environment.

- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) : GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots.

- [Asimov](https://github.com/asimovinc/asimov-v0) : open-source bipedal robotic legs

- [Training a humanoid robot for locomotion using Reinforcement Learning](https://github.com/rohanpsingh/LearningHumanoidWalking)

- [Reinforcement learning training code for AgiBot X1](https://github.com/AgibotTech/agibot_x1_train)

- [Simulation verification and physical deployment of robot reinforcement learning ](https://github.com/fan-ziqi/rl_sar)

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 

- [Safe RLHF](https://github.com/PKU-Alignment/safe-rlhf) : Constrained Value Alignment via Safe Reinforcement Learning from Human Feedback

---

# venv

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/Mac

```bash
python3 -m venv .venvsource 
.venv/bin/activate
```

### 3. Install everything

```bash
pip install -r requirements.txt
```

# Refrences

[Humanoid robot - Wikipedia](https://en.wikipedia.org/wiki/Humanoid_robot)

[What are Humanoid Robots and Why do They Matter? | NVIDIA Glossary](https://www.nvidia.com/en-in/glossary/humanoid-robot/)

[GitHub - YanjieZe/awesome-humanoid-robot-learning: A Paper List for Humanoid Robot Learning. · GitHub](https://github.com/YanjieZe/awesome-humanoid-robot-learning)

[AI Robots Got Shockingly Human This Year (2026 Update) - YouTube](https://www.youtube.com/watch?v=GjokTDha_vs)

https://interestingengineering.com/ai-robotics/9-humanoid-robots-at-ces-2026

https://humanoidroboticstechnology.com/articles/top-12-humanoid-robots-of-2026/

[reinforcement-learning-from-human-feedback](https://github.com/topics/reinforcement-learning-from-human-feedback)

[humanoid-robots : GitHub](https://github.com/topics/humanoid-robots)
