

### What is Humanoid Robot?

humanoid robots are characterized by their [anthropomorphic design](https://en.wikipedia.org/wiki/Anthropomorphism "Anthropomorphism").

- torso

- head

- two arms

- two legs

---

### Actuator

[Actuators](https://en.wikipedia.org/wiki/Actuator "Actuator") are the motors responsible for motion in the robot.

**Types:**

- **Electric**  - Small, less powerful. Hence it is common to use multiple electric actuators for a single joint in a humanoid robot.

- **Hydraulic**  - Higher power, bulky in size. One solution to counter the size issue is [electro-hydrostatic actuators](https://en.wikipedia.org/wiki/Electro-hydraulic_actuator "Electro-hydraulic actuator")

- **Pneumatic**  - Gas Based. As they inflate, they expand along the axis, and as they deflate, they contract.

---

### Control

To maintain balance during the walk a robot needs information about contact force and its current and desired motion. The solution to this problem relies on a major concept, the [Zero Moment Point](https://en.wikipedia.org/wiki/Zero_Moment_Point "Zero Moment Point") (ZMP).

Humanoids should be able to move in complex environments, planning and control must focus on self-collision detection, path planning and obstacle avoidance. I have written a editorial on [A-star algorithm](https://nimav.at/courses/2026-spring-cs405/editorials/a-star/) which helps in path planning and obstacle avoidance.

---

### History

**1961** : [Unimate](https://en.wikipedia.org/wiki/Unimate "Unimate") ; The first digitally operated and programmable non-humanoid robot.

**1972** : WABOT-1 ; world's first full-scale humanoid intelligent robot. It was the first android able to walk, communicate with a person in Japanese (with an artificial mouth), measure distances and directions to the objects using external receptors (artificial ears and eyes), and grip and transport objects with hands.

**1984** : WABOT-2 ; a musician humanoid robot able to communicate with a person, read a normal musical score with his eyes and play tunes of average difficulty on an electronic organ

**1985** : WHL-11 ; biped robot capable of static walking on a flat surface at 13 seconds per step and it can also turn.

**1986** : [Honda E series](https://en.wikipedia.org/wiki/Honda_E_series "Honda E series") ; early experimental biped humanoid robot series by Honda

**1993** : [Honda P series](https://en.wikipedia.org/wiki/Honda_P_series "Honda P series") ; advanced prototype humanoid robots with upper limbs

**1995** : [WABIAN](https://en.wikipedia.org/wiki/WABIAN "WABIAN") ; full-scale human-like walking robot from Waseda University

**1997** : Hadaly-2 ; interactive humanoid robot focused on human communication

**2000** : [ASIMO](https://en.wikipedia.org/wiki/ASIMO "ASIMO") ; iconic humanoid robot by Honda capable of walking and running

**2001** : [QRIO](https://en.wikipedia.org/wiki/QRIO "QRIO") ; entertainment humanoid robot developed by Sony

**2001** : HOAP ; Fujitsu’s compact humanoid research robot platform

**2002** : [HRP-2](https://en.wikipedia.org/wiki/HRP-2 "HRP-2") ; advanced biped humanoid robot for robotics research

**2003** : [Actroid](https://en.wikipedia.org/wiki/Actroid "Actroid") ; highly realistic android robot with silicone skin

**2004** : KHR-1 ; programmable hobby humanoid robot by Kondo Kagaku

**2005** : [HUBO](https://en.wikipedia.org/wiki/Hubo "HUBO") ; South Korean walking humanoid robot developed by KAIST

**2005** : PKD Android ; humanoid robot modeled after author Philip K. Dick

**2005** : Wakamaru ; domestic companion robot by Mitsubishi Heavy Industries

**2006** : [NAO](https://en.wikipedia.org/wiki/Nao_(robot) "NAO") ; small programmable humanoid robot widely used in education and research

**2006** : [iCub](https://en.wikipedia.org/wiki/ICub "iCub") ; open-source humanoid robot designed for cognition research

**2007** : TOPIO ; humanoid robot capable of playing table tennis

**2008** : Justin ; humanoid robot developed by the German Aerospace Center (DLR)

**2008** : Nexi ; social humanoid robot combining mobility, dexterity, and interaction

**2008** : Surena ; Iranian humanoid robot capable of speech and object tracking

**2009** : HRP-4C ; realistic Japanese humanoid robot capable of singing and dancing

**2009** : DARwIn-OP ; open-source humanoid robot platform for research and education

**2010** : [Robonaut 2](https://en.wikipedia.org/wiki/Robonaut "Robonaut 2") ; NASA humanoid robot designed for space missions

**2010** : REEM ; autonomous humanoid service robot by PAL Robotics

**2011** : ASIMO (2nd Generation) ; upgraded ASIMO with semi-autonomous capabilities

**2012** : NimbRo-OP ; open humanoid robot platform for autonomous robotics research

**2013** : TORO ; torque-controlled humanoid robot developed by DLR

**2013** : [Valkyrie (R5)](https://en.wikipedia.org/wiki/Valkyrie_(robot) "Valkyrie (R5)") ; NASA humanoid robot for future space exploration missions

**2014** : Manav ; India’s first 3D-printed humanoid robot

**2014** : [Pepper](https://en.wikipedia.org/wiki/Pepper_(robot) "Pepper") ; social humanoid robot designed for public interaction

**2014** : Nadine ; socially intelligent humanoid robot modeled after a real person

**2016** : [Sophia](https://en.wikipedia.org/wiki/Sophia_(robot) "Sophia") ; AI-powered humanoid robot known for realistic conversations

**2016** : OceanOne ; underwater humanoid robot with haptic feedback capabilities

**2017** : TALOS ; industrial humanoid robot for advanced manipulation tasks

**2018** : Rashmi Robot ; Indian multilingual humanoid robot with emotional interaction

**2020** : [Digit](https://en.wikipedia.org/wiki/Digit_(robot) "Digit") ; humanoid robot for logistics and autonomous delivery tasks

**2020** : Vyommitra ; Indian humanoid robot developed for the Gaganyaan space mission

**2020** : Robot Shalu ; multilingual Indian humanoid robot built using recycled materials

**2022** : [Ameca](https://en.wikipedia.org/wiki/Ameca_(robot) "Ameca") ; highly expressive humanoid robot by Engineered Arts

**2022** : [Optimus](https://en.wikipedia.org/wiki/Optimus_(robot) "Optimus") ; Tesla’s general-purpose humanoid robot initiative

**2024** : [Atlas (Electric)](https://en.wikipedia.org/wiki/Atlas_(robot) "Atlas (Electric)") ; fully electric humanoid robot by Boston Dynamics

**2024** : G1 ; affordable humanoid robot by Unitree Robotics

**2024** : HumanPlus ; humanoid robot capable of mimicking human movements

**2025** : SE01 ; humanoid robot capable of performing forward flips

**2025** : Figure 03 ; autonomous humanoid robot designed for household chores

**2025** : NEO ; consumer-ready humanoid robot for home assistance

**2025** : AgiBot A2 ; humanoid robot that set a long-distance walking world record

---

### Zero moment point

ZMP is the point on the ground where the robot’s total tipping moment becomes zero.

If the ZMP stays inside the support area (area covered by the feet touching the ground), the robot remains stable.

If the ZMP moves outside the support area, the robot starts to tip or fall.

![The ZMP is an axis, not a point](https://scaron.info/images/cmp-zmp-popovic-herr.png)

---

## Refrences

[Humanoid robot - Wikipedia](https://en.wikipedia.org/wiki/Humanoid_robot)

[GitHub - YanjieZe/awesome-humanoid-robot-learning: A Paper List for Humanoid Robot Learning. · GitHub](https://github.com/YanjieZe/awesome-humanoid-robot-learning)




