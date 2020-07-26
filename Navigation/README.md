# Navigation
#### DQN implementation for training an agent to navigate and collect yellow bananas while avoiding blue bananas in a square world.

##### Environment Description:
**A reward of +1** : collecting a "yellow" banana
**A reward of -1** : collecting a "blue" banana

**State space has 37 dimensions** and contains the agent's velocity, alongwith the ray-based perception of objects around the agent's forward direction.

**Discrete Action Space:**
0 - move forward
1 - move backward
2 - turn left
3 - turn right

**An average reward of +200** is considered solving the environment.
##### Platform: 
``` bash 
Debian 10
```

##### Python version: 
``` python
python 3.7.3
```

##### Library:
```python
pytorch
```

##### Installation Steps:
```python
# Clone drlnd from https://github.com/udacity/deep-reinforcement-learning
# Create a conda environment
conda create drlnd
source activate drlnd
```
```bash 
# move to the python folder in the drlnd repo
cd python
pip3 install . 
```
This installs all the dependencies.
```bash 
# Clone this repo and copy the env folder from this repo into the p1-navigation of the drlnd repo.
jupyter-lab --port=8888
```
Run the [**Navigation_DQN**][nav notebook] notebook.
Find the **project report** [**here**][report]. 

##### Author:
[Amal Patel](https://www.linkedin.com/in/patelamalk/)

[//]: # (Use this part to save the links and use the references)

[nav notebook]:<https://github.com/patelamalk/RL/blob/master/Navigation/Navigation_DQN.ipynb>
[report]:<https://github.com/patelamalk/RL/blob/master/Navigation/Report.md>


