"""
Implementation of the replay buffer
"""
import numpy as np
from collections import deque, namedtuple
import random
import torch

class ReplayBuffer:

    def __init__(self, buffer_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """
        :param state  (numpy array)     : state
        :param action (scalar)          : action in the current state
        :param reward (scalar)          : reward after taking an action in current state
        :param next_state(numpy array)  : state reached after taking action in the current state
        :param done (bool)              : if the episode terminated
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, mini_batch_size):
        """
        :return: A sample of minibatch from the memory
        """
        mini_batch = random.sample(self.memory, mini_batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in mini_batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in mini_batch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in mini_batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in mini_batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in mini_batch if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
