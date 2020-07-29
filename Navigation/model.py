import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Extra layer have been added i.e. the 3rd fully connected layer.
    Input -> fc1 -> fc2 -> fc3 -> # actions
    Note :
        batch normalization could be applied
        Or implement prioritized experience replay
    """

    def __init__(self, state_size, action_size, fc1_units, fc2_units, fc3_units):
        """
        :param state_size: state space dimension
        :param action_size: action space dimension
        :param fc1_units: # of fully connected nodes in the 1st layer
        :param fc2_units: # of fully connected nodes in the 2nd layer
        :param fc3_units: # of fully connected nodes in the 3rd layer
        """
        super(DQN, self).__init__()

        self.l1 = nn.Linear(state_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, fc3_units)
        self.l4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)
