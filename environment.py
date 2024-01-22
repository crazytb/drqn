import numpy as np
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box, MultiBinary
import gymnasium as gym
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from numpy.random import default_rng


# Parameters
POWERCOEFF = 0.1


class PNDEnv(Env):
    def __init__(self, n=10):
        super(PNDEnv, self).__init__()
        self.n = n
        # Actions we can take 0) transmit and 1) listen
        self.action_space = MultiBinary(self.n)
        # Observation space
        self.observation_space = spaces.Dict({
            "tx_prob": Box(low=0, high=1, shape=(self.n, 1)),
            "prev_result": MultiBinary([self.n, 2]),
            # [1, 0, 0, 0]: Transmission,
            # [0, 1, 0, 0]: Listen and Channel Idle,
            # [0, 0, 1, 0]: Listen and No Collision, 
            # [0, 0, 0, 1]: Listen and Collision Detected,
        })

    def _get_obs(self):
        transmission_prob = np.reshape(self._px_prob, newshape=(self.n))
        prev_result = np.reshape(self._prev_result, newshape=(2*self.n))
        return np.concatenate([transmission_prob, prev_result])
    
    def _get_info(self):
        print("Transmission Probability, Prev Result")
        for i in range(self.n):
            print(f"Node {i}: {self._px_prob[i]}, {self._prev_result[i]}")
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # State reset
        self._px_prob = np.random.uniform(0, 1, size=(self.n, 1))
        self._prev_result = np.tile([0, 1], reps=(self.n, 1))
        observation = self._get_obs()
        info = None
        return observation, info

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
