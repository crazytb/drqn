import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from numpy.random import default_rng

# Hyperparameters
learning_rate = 0.0001
gamma = 1

# Parameters
POWERCOEFF = 0.1


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, round(n_observations/2))
        self.layer2 = nn.Linear(round(n_observations/2), round(n_observations/2))
        self.layer3 = nn.Linear(round(n_observations/2), n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

class DRQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DRQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, round(n_observations/2))
        self.layer2 = nn.LSTM(round(n_observations/2), round(n_observations/2))
        self.layer3 = nn.Linear(round(n_observations/2), n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.softmax(x)


# class OneHotEncodedSpace(gym.Space):
#     """
#     {0,...,1,...,0}

#     Example usage:
#     self.observation_space = OneHotEncoding(size=4)
#     """
#     def __init__(self, size=None):
#         assert isinstance(size, int) and size > 0
#         self.size = size
#         gym.Space.__init__(self, (), np.int64)

#     def sample(self):
#         one_hot_vector = np.zeros(self.size)
#         one_hot_vector[np.random.randint(self.size)] = 1
#         return one_hot_vector

#     def contains(self, x):
#         if isinstance(x, (list, tuple, np.ndarray)):
#             number_of_zeros = list(x).contains(0)
#             number_of_ones = list(x).contains(1)
#             return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
#         else:
#             return False

#     def __repr__(self):
#         return "OneHotEncoding(%d)" % self.size

#     def __eq__(self, other):
#         return self.size == other.size
    
#     def __len__(self):
#         return self.size


class PNDEnv(Env):
    def __init__(self):
        super(PNDEnv, self).__init__()
        # Actions we can take 0) transmit and 1) listen
        self.action_space = Discrete(2)
        # Observation space
        self.observation_space = spaces.Dict({
            "tx_prob": spaces.Box(low=0, high=1, shape=(1, 1)),
            "prev_result": spaces.MultiBinary(4),
            # [1, 0, 0, 0]: Transmission,
            # [0, 1, 0, 0]: Listen and Channel Idle,
            # [0, 0, 1, 0]: Listen and No Collision, 
            # [0, 0, 0, 1]: Listen and Collision Detected,
        })
        self._neighbors = None

    def _get_obs(self):
        return {"transmission_prob": self._px_prob,
                "prev_result": self._prev_result}
        
    def reset(self, seed=None, neighbors=None):
        super().reset(seed=seed)
        # State reset
        self._px_prob = np.random.uniform(0, 1, size=(1, 1))
        self._prev_result = np.array([0, 0, 0, 1])
        self._neighbors = neighbors
        observation = self._get_obs()
        info = None
        return observation, info

    def _act2act(self, action):
        if self._neighbors == None or self._neighbors == 0:
            raise ValueError("Cannot perform action when self._neighbors == 0")
        else:
            if action == 0:
                return 0
            else:
                self._neighbors_tx_event = np.random.rand(self._neighbors) < (1/self._neighbors)
                if np.sum(self._neighbors_tx_event) == 0:
                    return 1
                elif np.sum(self._neighbors_tx_event) == 1:
                    return 2
                else:
                    return 3
            

    def step(self, action):  # 여기 해야 함.
        action = self._act2act(action)
        if action == 0:
            self._prev_result = np.array([1, 0, 0, 0])
            if np.sum(self._neighbors_tx_event) == 0:
                reward = 1
            else:
                reward = -POWERCOEFF
        elif action == 1:
            self._prev_result = np.array([0, 1, 0, 0])
            reward = 0
        elif action == 2:
            self._prev_result = np.array([0, 0, 1, 0])
            reward = 0
        else:
            self._prev_result = np.array([0, 0, 0, 1])
            reward = 0
        
        done = False
        observation = self._get_obs()

        return observation, reward, False, done, self.info

    
    def render(self):
        # Implement viz
        pass
