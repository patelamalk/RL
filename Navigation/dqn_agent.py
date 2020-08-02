import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed):
        # Initialize the state size, action size and seed
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Initialize the local and target network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Initialize the replay buffer for experiences
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step for learning every n time steps i.e. updating the weights from the local network onto the target network
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Add the experience to the memory
        self.memory.add(state, action, reward, next_state, done)
        # If the time step is 4 then the weights are updated
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        # Take an action based on the value of epsilon
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Get the action values from the local network
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        Q_targets_next_indicies = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_targets_next_indicies)
        # One step look ahead guesses for the action values
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Current guess
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Update the guess in the direction of the expected guess i.e. expected Q values
        loss = F.mse_loss(Q_expected, Q_targets)
        # Remove the old gradients and only back propagate the current gradients wrt the loss here
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        # transfer the weights from the local network onto the target network        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        # Initialize the memmory of given buffer size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        # Append experiences
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        # Sample experiences from the buffer of mini batch sizes
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # Return the len of the buffer
        return len(self.memory)