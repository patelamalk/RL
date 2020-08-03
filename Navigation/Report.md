### Dueling DQN implementation for training an agent to navigate and collect yellow bananas while avoiding blue bananas in a square world

### Environment Description

- There are **37 dimensions** of the state space.
- There are **4 discrete** actions for the environment.
- An **average reward of +13** is considered solving this environment.

### Implementation Details

**The learning is not directly from pixels. The visual observation is not utilized as state space.**    
**The [DQN][Attempt1] implementation for this problem didn't converge under 1800 episodes**      
The **Dueling DQN** converged in **495** episodes.

#### Dueling DQN

- This [paper][Dueling] was used for implementation of Dueling DQN.

- We split the traditional Q(s, a) from the DQN into 2 components i.e. A(s, a) and V(s)
- One example, the agent could learn from the state function if a particular state is good to be in and requires taking no actions.
- **Linear(state_size, 64) -> ReLU() -> Linear(64, 64) -> ReLU() -> Linear(64, 1)**[State value]**         - (i)**   
- **Linear(state_size, 64) -> ReLU() -> Linear(64, 64) -> ReLU() -> Linear(64, 4)**[Advantage function]**  - (ii)**
- **Return  V(s) + (A(s, a) - mean(A(s, a)))**
- This represents the Q(s, a) nn the output node of the network.  
![Network Architecture][nw]

#### Replay Buffer Class

- Used a deque(circular linked list) to store the state, action, reward, next_state, done.    
- Implemented add method to sequentially add experiences.     
- Implemented sample method to sample minibatches of experiences(64).      

#### Agent Class

- Initialize the local and target networks.
- Initialize the memory.
- Implemented step method, agent adds the experience and learns every 4 steps in an episode.
- Implemented act method, agent selects action for a state with the epsilon greedy policy.
- Implemented learn method, agent optimizes the local network.
- Implemented soft update, transfer the weights of the lcoal network on the target network.

#### Hyperparameters

|Parameter| Value|
|---------|------|
|BUFFER_SIZE|1e5|
|BATCH_SIZE|64|
|GAMMA|0.99|
|LR|5e-4|
|UPDATE_EVERY|4|
|TAU|1e-3|

### Why does Dueling DQN work?

- The Dueling architecture breaks down the Q into 2 parts, A(advantage) and V(state value)
- The agent is learning the state value and the best action for a given state.
- In many states, the value of different actions may be similar and there is no real importance on action chosen in the given state.
- In the traditional DQN the updates are applied only to the specific action taken in a state and thus the learning is slow.
- In contrast, the agent in each training iteration learns the state value even though it had taken only a single action in a given state.

### Algorithm

- Initialize the agent
- Initialize scores
- for episodes <- 1 to n_episodes:
  - reset the environment.
  - for time_step <- 1 to n_step:
    - take step with epsilon-greedy action.
    - prepare (state, action, reward, next_state, done).
    - Agent takes a step i.e. learns from the experience, update weights.
  - Calculate the average reward for the episode
- plot the reward

### Rewards plot

![Avg. Reward / Episode][reward plot]

### Future Work

The following techniques could help reduce overfitting.
- Hyperparameter tuning.
- L2 Regularization and Dropout for neural networks.   

Switching the algorithms can also reduce the number of episodes required to converge
- [Prioritized Expereince Replay][per]
- [Asynchronous Reinforcement Learning][async rl]


### References
- [Deep mind atari][deepmind atari]
- [Prioritized Experience Replay][per]
- [Asynchronous Reinforcement Learning][async rl]
- [Dueling DQN][Dueling]
- [stackoverflow][sf]
- [Udacity DRLND repository][udacity]

[//]: # (Create alias for all hyperlinks here, no formatting required)

[deepmind atari]:<https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf>

[per]:<https://arxiv.org/abs/1511.05952>

[async rl]:<https://arxiv.org/pdf/1602.01783.pdf>

[Dueling]:<https://arxiv.org/pdf/1511.06581.pdf>

[reward plot]:<https://github.com/patelamalk/RL/blob/master/Navigation/plots/Rewards.png?raw=True>

[Attempt1]:<https://github.com/patelamalk/RL/tree/master/Navigation/Attempt%201%20DQN>

[sf]:<https://datascience.stackexchange.com/questions/34074/dueling-dqn-cant-understand-its-mechanism?rq=1>

[udacity]:<https://github.com/udacity/deep-reinforcement-learning>

[nw]:<https://github.com/patelamalk/RL/blob/master/Navigation/plots/Network_Architecture.png?raw=True>

[L2]:<https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261>

[per implmentation]:<https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/>