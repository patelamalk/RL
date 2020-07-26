### Aim
- To train an Agent to navigate and collect yellow bananans while avoiding the blue bananas in the square world.  

### Environment Description
- There are **37 dimensions** of the state space.
- There are **4 discrete** actions for the environment.
- An **average reward of 200** is considered solving this environment.

### Implementation Details
**The learning is not directly from pixels. The visual observation is not utilized as state space.**
##### DQN
- Implemented [**DQN**][deepmind atari]         
- It uses 2 deep neural networks, local and target as mentioned in [this paper.][deepmind atari]

|Layer|# of units|   
|-----|-------|  
|Input (states)|37, 1, batch_size|
|Fully connected|128|
|Fully connected|128|
|Output (actions)| 4|

- The local network is the policy network utilized by the agent to decide actions.      
- The target network is used to compute the one step look ahead values i.e. the expected next_state value.      

##### Replay Buffer Class
- Used a deque(circular linked list) to store the state, action, reward, next_state, done.    
- Implemented add method to sequentially add experiences.     
- Implemented sample method to sample minibatches of experiences(64).      

##### Agent Class
- Initialize the local and target networks.
- Initialize the memory.
- Implemented step method, agent adds the experience and learns every 5 steps in an episode.
- Implemented act method, agent selects action for a state with the epsilon greedy policy.
- Implemented learn method, agent optimizes the Q function i.e. local network.
- Implemented soft update, transfer the weights of the lcoal network on the target network.

##### Hyperparameters

|Parameter| Value|
|---------|------|
|BUFFER_SIZE|10000|
|BATCH_SIZE|64|
|GAMMA|0.99|
|LR|5e-3|
|UPDATE_EVERY|5|
|TAU|1e-3|

### Algorithm
- Initialize the agent
- Initialize scores
- for episodes <- 1 to n_episodes:
    - reset the environment.
    - for time_step <- 1 to n_step:
        * take step with epsilon-greedy action.
        * prepare (state, action, reward, next_state, done).
        * Agent takes a step i.e. learns from the experience, update weights.
    - Calculate the average reward for the episode
- plot the reward

### Rewards plot
![Avg. Reward / Episode][reward plot]

### Future Work
- The agent has not solved the environment.
- The implementation could be improved by using Dueling DQN, Prioritized Replay.
- Hyperparameter tuning.

[//]: # (Create alias for all hyperlinks here, no formatting required)

[deepmind atari]:<https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf>
[tmp image]:<https://icatcare.org/app/uploads/2018/07/Thinking-of-getting-a-cat.png>
[reward plot]:<https://github.com/patelamalk/RL/blob/master/Navigation/plots/Reward.png?raw=True>
