from model import DQN
from ReplayBuffer import ReplayBuffer

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

class Agent():

    def __init__(self, state_size, action_size, device, buffer_size, batch_size, gamma, tau, lr, update_every, fc1, fc2, fc3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Initialize the nodes for the fully connected layers in the network
        self.fc1_units = fc1
        self.fc2_untis = fc2
        self.fc3_units = fc3

        # Initialize the hyperparameters
        self.BUFFER_SIZE = buffer_size      # Replay buffer size
        self.BATCH_SIZE = batch_size        # minibatch size
        self.GAMMA = gamma                  # discount factor
        self.TAU = tau                      # soft update of the parameters
        self.LR = lr                        # learning rate
        self.UPDATE_EVERY = update_every    # how often to update the network

        # Initialize the local and target network
        self.local_nw = DQN(self.state_size, self.action_size, self.fc1_units, self.fc2_untis, self.fc3_units).to(self.device)
        self.target_nw = DQN(self.state_size, self.action_size, self.fc1_units, self.fc2_untis, self.fc3_units).to(self.device)

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.local_nw.parameters(), lr=self.LR)

        # Initialize the replay buffer
        self.memory = ReplayBuffer(self.BUFFER_SIZE, self.device)

        # Initialize the time step
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        # Increment the time step parameter, to update it every 4 steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample(self.BATCH_SIZE)

    def act(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_nw.eval()
        with torch.no_grad():
            action_values = self.local_nw(state)
        self.local_nw.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # Max predicted Q value from the target model
        Q_targets_next = self.local_nw(next_states).detach().max(1)[1].unsqueeze(1)
        # Q target for the current states
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Expected Q values from the local model
        Q_expected = self.local_nw(states).gather(1, actions)

        # Compute the loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.local_nw, self.target_nw, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        0_target = tau*0_local + (1-tau)*)_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau)*target_param.data)