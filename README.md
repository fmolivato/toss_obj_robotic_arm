# Object Toss with a Robotic Arm

<p align="center" width="100%">
<img height="300" src="images/throwing.gif">
</p>

__Table of Contents:__
- [Introduction](#introduction)
- [Reinforcement Learning Laboratory](#reinforcement-learning-laboratory)
- [Inspiration](#inspiration)
- [Development Tools](#development-tools)
- [Custom Env](#custom-env)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward](#reward)
  - [Vectorized Environment](#vectorized-environment)
- [Learning Paradigm](#learning-paradigm)
  - [RL](#rl)
  - [Deep RL](#deep-rl)
  - [Policy Gradient Methods](#policy-gradient-methods)
  - [Proximal Policy Optimization](#proximal-policy-optimization)
  - [Curriculum Learning](#curriculum-learning)
- [Results](#results)
- [Solution Analysis](#solution-analysis)
- [Dev Challenges](#dev-challenges)
- [Further Considerations](#further)
  - [Vision Control](#vision-control)
  - [Automatic Curriculum Learning](#automatic-curriculum-learning)
  - [Parallel Multitask RL](#parallel-multitask-rl)
  - [Sim to Real](#sim-to-real)

## Introduction
This project was made possible thanks to the Reinforcement Learning laboratory that is part of the Master's course in "Artificial Intelligence Engineering" at the University of Modena and Reggio Emilia. 
The laboratory was made possible thanks to Professor Simone Calderara, who managed and coordinated its development.

## Reinforcement Learning Laboratory
During the laboratory, it was possible to delve into the fundamentals of Reinforcement Learning and their implementation. We used various algorithms, both value-based and policy-based. The simpler ones, such as Q-learning and Deep Q-learning, were implemented from scratch, while for the more complex theoretical ones, like A2C and PPO, we chose to use the implementations from "Stable Baseline 3." We also utilized environments like Gymnasium, Petting-zoo, and Pybullet. Furthermore, we developed some custom environments, one of which was used in this project. Below, you can see some of our implementations:

<p align="center" width="100%">
<img height="300" src="images/RL_gif.gif">
</p>

For the sake of completeness, we provide the materials upon which our studies were based, with the hope that they will serve as a useful reference for all readers to better understand and delve into Reinforcement Learning:
1. [Fundamentals of Reinforcement Learning](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning)
2. [Deep Mind Introduction to Reinforcement Learning (1)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
3. [Deep Mind Introduction to Reinforcement Learning (2)](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)
4. [Deep Mind Introduction to Reinforcement Learning (3)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)
5. [Reinforcement Learning Italian Lectures](https://www.youtube.com/playlist?list=PLMee1hSjLKdBymYS-wBYuKdQOuwQbnqdb)
6. [Spinning up](https://spinningup.openai.com/en/latest/index.html)
7. [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
8. [Sutton and Barto RL Book](http://incompleteideas.net/book/RLbook2020.pdf)

## Inspiration
This work is inspired by _[TossingBot: Learning to Throw Arbitrary Objects with Residual Physics (Princeton University, Google, Columbia University, MIT.)](https://tossingbot.cs.princeton.edu/)_ (below).

<p align="center" width="100%">
<img src="images/insp.gif">
</p>

In order to make the task a little easier from the exploration point of view, we replaced the fingers with a plate. In this setting the robot need to learn to throw an object using just a flat surface.
## Development Tools

 1. Mujoco
 2. Gymansium-Robotics
 3. StableBaselines3

<!-- This work was developed as a course project of Smart Robotics, University of Modena and Reggio Emilia, Italy

<p align="center" width="100%" >
<a href="https://www.canva.com/design/DAFspXKsex4/6PJ41YdfBPMxjuhkqJpZkw/view?utm_content=DAFspXKsex4&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink">:bar_chart: Here the slides of # the presentation :bar_chart:</a>
</p> -->

## Custom Env

### Observation Space

<p align="center" width="100%">
<img height="300" src="images/frame.png">
</p>

There are 19 observations:

+ Shoulder rotation (rad)
+ Elbow rotation  (rad)
+ Wrist rotation (rad)
+ Shoulder angular velocity (rad/s)
+ Elbow angular velocity (rad/)
+ Wrist angular velocity (rad/s)
+ Wrist coordinate (m)
+ Object coordinate (m)
+ Goal coordinate (m)

__Observation range:__ [-inf, inf]

### Action Space
<p align="center" width="100%">
<img height="300" src="images/jont.png">
</p>

__Actions:__
+ Shoulder joint 2 DoF
+ Elbow joint 1 DoF
+ Wrist joint  2 DoF

__Control range:__ [-2, 2] (N·m)

### Reward

$Reward = - \lVert obj\\_pos - target\\_pos \rVert - 0.1 * \lVert action \rVert ^2 $

The first term force the learning process to focus on throwing the object effectively. The second term force the learning process to develop an efficient motion of the robotic arm. The 0.1 weight scales down the second term in order to delay the learning process of the efficient motion only after being able to solve the main task.

### Vectorized Environment

<p align="center" width="100%">
<img height="300" src="images/vec-train-step-19200-to-step-19400_out0001.jpg">
</p>

Allows multiple instances of the same environment to run in parallel which leds to a more efficient utilization of computing resources. 
This settings enhance exploration, as the agent can explore different parts of the state space simultaneously. Every instance has its own separate state.

We used 16 parallel envs during the training.

## Learning Paradigm
### RL
<p align="center" width="100%">
<img height="300" src="images/rl.png">
</p>

It is the third paradigm of Machine Learning. In this setting the agent learns to take actions by exploring the environment, observes outcomes, and adjust its strategy (policy) to maximize total rewards.

### Deep RL
<p align="center" width="100%">
<img height="300" src="images/deep_rl.png"/>
</p>

Specific approach within RL that uses deep neural networks to approximate complex decision-making functions i.e. handle high-dimensional and intricate state and action spaces.

### Policy Gradient Methods
This methods aim to directly learns a policy to solve the given task.

<p align="center" width="100%">
<img height="300" src="images/pgm_exp.jpg">
</p>

where:
+ $A_t$ is called 'Advantage' and is defined as $A_t = G_t - V_t$
+ $G_t$ is the return i.e. the discounted sum of rewards in a single episode
+ $V_t$ is the value function i.e. the estimation of $G_t$ done at time $t$

### Proximal Policy Optimization
The Policy Gradient Methods is calculated online i.e. the policy is optimized on an observation history sampled by the policy itself. This leads to major instabilities in the learning process which sometimes brings the policy to diverge by focusing on a constantly shifting distribution. When it happens, the agent is not working towards the main goal anymore and thus stops to learn. To solve this issue OpenAI engineered the Proximal Policy Optimization (PPO).
Basically, the new policy gradient method force the agent to learn a "proximal" policy which is a policy not too different from the one in the previous episode. The plots below, shows how they are clipping the loss in order to avoid extreme changes on the main strategy.

<p align="center" width="100%">
<img src="images/ppo-surrogate.jpg">
</p>
<p align="center" width="100%">
<img height="300" src="images/ppo_plot.png">
</p>

### Curriculum Learning

<p align="center" width="100%">
<img height="300" src="images/cur1.png">
</p>

The curriculum learning is the concept of decomposing a complex task in simpler handcrafted subtasks in order to effectively reach the overall goal.

We investigated this approach in order to teach the agent to both throw the object and also track the target.
Indeed the fist curriculum task was to learn the throwing motion to a static target. Instead the next two curriculums tasks focused on the tracking skill by gradually increasing the spawn offset of the target.

We trained the agent for 12M steps over each task.

## Results
<p align="center" width="100%">
<img height="300" src="images/results.gif">
</p>

We realized that in order to solve the task, the agent learned to "slap" the object. It resambles the behavior of the human hand during a basketball shot.


## Solution Analysis
+ Not requires any explicit model of the kinematics, neither direct nor inverse. __This big plus when dealing with very complex dexterous robots__.
+ Not requires any model of the dynamics, neither direct nor inverse.
+ No explicit control model.
+ Not requires multiple control models for different interactions patterns (like interaction and interaction free tasks). Just one comprehensive control.
+ Online adaptation to variations (e.g. The mathematical model of electric actuations will change with time).
+ Scalable fast training on complex tasks thanks to vectorized envs.
+ Reduce entry level skills of the operator.
+ Short time to deploy.
  
## Dev Challenges
+ Broken video recorder lib of GPU parallel virtualized environment.
+ Bad documentation of the Mujoco Lib
+ Bad documentation of the StableBaseline3 regarding curriculum.
+ Different base env class between tasks. Very library dependent. It makes hard the integration of a custom pipeline.

## Further
This section describes some considerations we made to further improve the flexibility and the effectiveness of this learning approach.

### Vision Control
We realized that introducing a vision control feedback, this solution could be adopted more easily in a real enviroment.

+ __End-to-End Traning:__ The observations are entextracted by a Neural Network instead of retreiving them from the simulator API.

### Automatic Curriculum Learning

<p align="center" width="100%">
<img src="images/acl.png">
</p>

The Automatic Curriculum Learning (ACL) is a process that identify automatically the correct sequential task used to train the agent, instead of defining them by hand.

### Parallel Multitask RL

<p align="center" width="100%">
<img height="300" src="images/parallel.png">
</p>

The Parallel Multitask RL is a training pattern where multiple pretext tasks are trained togheter in order to share useful knowledge for the main task. It is similar to ACL(above) but the tasks are solved in parallel and not sequentially.

### Sim to Real

<p align="center" width="100%">
<img height="300" src="images/dextreme.jpg">
</p>

Honorable mention to _[DeXtreme: Transfer of Agile In-Hand Manipulation from Simulation to Reality](https://dextreme.org/)_ a work made by NVIDIA. This work proves that is possible to go from simulation to reality effictively even on a complex task like robotic hand dexterity.
