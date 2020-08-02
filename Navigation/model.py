import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Implementing duel DQN.
    Refer https://github.com/patelamalk/RL/blob/master/Navigation/Report.md#dueling-dqn for a more detailed explanation.
    """    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Represents V(s) i.e.state value
        self.state_value = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )
        # Represents A(s, a) i.e. advantage function
        self.advantage = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        """
        return: Returns the eqn, V(s) + [A(s,a) - mean(A(s,a))]
        """
        state_value = self.state_value(state)
        advantage = self.advantage(state)
        return state_value + (advantage - advantage.mean())