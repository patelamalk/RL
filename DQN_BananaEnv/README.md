# Navigation      

### Dueling DQN implementation for training an agent to navigate and collect yellow bananas while avoiding blue bananas in a square world.

### Environment Description:
**__A reward of +1__** : collecting a "yellow" banana         
**A reward of -1** : collecting a "blue" banana

**State space has 37 dimensions** and contains the agent's velocity, alongwith the ray-based perception of objects around the agent's forward direction.

**Discrete Action Space:**     
0 - move forward           
1 - move backward           
2 - turn left           
3 - turn right         

**An average reward of +13** is considered solving the environment.            

### Platform: 
``` bash 
Debian 10
```

### Python version: 
``` python
python 3.7.3
```

### Library:
```python
pytorch
```

### Installation Steps:
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

Run the [**Bananan_Duel_DQN.ipynb**][nav notebook] notebook.             
Find the **project report** [**here**][report].          
    
### Author:
[Amal Patel](https://www.linkedin.com/in/patelamalk/)

### References
- [Deep mind atari][deepmind atari]
- [Prioritized Experience Replay][per]
- [Asynchronous Reinforcement Learning][async rl]
- [Dueling DQN][Dueling]
- [stackoverflow][sf]
- [Udacity DRLND repository][udacity]

[//]: # (Use this part to save the links and use the references)

[nav notebook]:<https://github.com/patelamalk/RL/blob/master/DQN_BananaEnv/Banana_Duel_DQN.ipynb>

[report]:<https://github.com/patelamalk/RL/blob/master/DQN_BananaEnv/Report.md>

[deepmind atari]:<https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf>

[per]:<https://arxiv.org/abs/1511.05952>

[async rl]:<https://arxiv.org/pdf/1602.01783.pdf>

[Dueling]:<https://arxiv.org/pdf/1511.06581.pdf>

[reward plot]:<https://github.com/patelamalk/RL/blob/master/Navigation/plots/Rewards.png?raw=True>

[Attempt1]:<https://github.com/patelamalk/RL/tree/master/Navigation/Attempt%201%20DQN>

[sf]:<https://datascience.stackexchange.com/questions/34074/dueling-dqn-cant-understand-its-mechanism?rq=1>

[udacity]:<https://github.com/udacity/deep-reinforcement-learning>
