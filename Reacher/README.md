# Continuous Control - Reacher Environment

#### DDPG for solving the reacher environment

#### Environment Description
A **reward of +0.1** is provided for each step, where the agent's hand is in the goal location
The **state space is comprised of 33 variables**; corresponding to position, rotation, velocity & angular velocities
**Action space is a vector of 4 numbers** having an entry b/w **+1 and -1**
**Solving : +30 avg. reward over 100 episodes**
![Reacher](./reacher.gif)

#### Platform
```bash
Ubuntu 16.04
```

#### Python version
```python
python 3.7.3
```

#### Library
```bash
unityagents
pyTorch
```

#### Installation Steps
```
virtualenv torch_reacher
source torch_reacher/bin/activate
pip3 install unityagents        
```
**Doing this in a new virtualenv will automatically install the compatible versions of all packages**
**Within an existing env, check the compatible versions of packages**

Run [**DDPG.ipynb**][ddpg]
Find the project report [**here**][report]

#### 

[//]: # (Use this to save the links and use the references)

[DS]:<http://proceedings.mlr.press/v32/silver14.pdf>
[JV]:<https://julien-vitay.net/deeprl/DeepRL.pdf>
[PE]:<http://proceedings.mlr.press/v32/silver14.pdf>
[ddpg]:<>
[report]:<>