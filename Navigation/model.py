import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.state_value = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        state_value = self.state_value(state)
        advantage = self.advantage(state)
        return state_value + (advantage - advantage.mean())