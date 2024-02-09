import sys
import collections
from typing import Dict, List, Tuple
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
else:
    device = torch.device("cpu")

# Parameters
POWERCOEFF = 0.1

class PNDEnv(Env):
    def __init__(self, n: int=10, density: float=0.1, model: str=None):
        super(PNDEnv, self).__init__()
        self.n = n
        self.density = density
        self.model = model
                
        # Actions we can take 0) transmit and 1) listen
        self.action_space = MultiBinary(self.n)
        # Observation space
        self.observation_space = spaces.Dict({
            "tx_prob": Box(low=0, high=1, shape=(self.n, 1)),
            "prev_result": MultiBinary([self.n, 1]),
            # [1, 0, 0, 0]: Transmission,
            # [0, 1, 0, 0]: Listen and Channel Idle,
            # [0, 0, 1, 0]: Listen and No Collision, 
            # [0, 0, 0, 1]: Listen and Collision Detected,
        })

    def _get_obs(self):
        transmission_prob = np.reshape(self._tx_prob, newshape=(self.n))
        prev_result = np.reshape(self._prev_result, newshape=(self.n))
        return np.concatenate([transmission_prob, prev_result])
    
    def _get_info(self):
        print("Transmission Probability, Prev Result")
        for i in range(self.n):
            print(f"Node {i}: {self._tx_prob[i]}, {self._prev_result[i]}")
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # State reset
        self._tx_prob = np.random.uniform(0, 1, size=(self.n, 1))
        self._prev_result = np.zeros((self.n, 1))
        self.adjacency_matrix = self._make_adjacency_matrix()  # Adjacency matrix
        self.remaining_nodes = [1]*self.n  # Remaining nodes
        
        observation = self._get_obs() 
        info = None
        return observation, info

    def step(self, action: np.array):  # 여기 해야 함.
        # Check if the action is valid. Action length must be equal to the number of nodes and action must be 0 or 1. 
        reward = 0
        assert len(action) == self.n, "Action length must be equal to the number of nodes."
        assert all([a in [0, 1] for a in action]), "Action must be 0 or 1."
        
        self._prev_result = action
        
        for i in range(self.n):
            if action[i] == 1:
                # If one of the adjacent nodes is transmitting, collision occurs. Reward is negative POWERCOEFF.
                if 1 in self.adjacency_matrix[i][action==1]:
                    reward -= POWERCOEFF
                # If one of the adjacent nodes is transmitting, collision occurs. Reward is negative POWERCOEFF.
        
        done = False
        observation = self._get_obs()

        return observation, reward, False, done, self.info

    
    def render(self):
        # Implement viz
        pass
    
    def _make_adjacency_matrix(self) -> np.ndarray:
        """Make adjacency matrix of a clique network.
        
        Args:
            n (int): Number of nodes.
            density (float): Density of the clique network.
        
        Returns:
            np.ndarray: Adjacency matrix.
        """
        if self.density < 0 or self.density > 1:
            raise ValueError("Density must be between 0 and 1.")
        
        n_edges = int(self.n * (self.n - 1) / 2 * self.density)
        adjacency_matrix = np.zeros((self.n, self.n))

        if self.model == "dumbbell":
            adjacency_matrix[0, self.n-1] = 1
            adjacency_matrix[self.n-1, 0] = 1
            for i in range(1, self.n//2):
                adjacency_matrix[0, i] = 1
                adjacency_matrix[i, 0] = 1
            for i in range(self.n//2+1, self.n):
                adjacency_matrix[i-1, self.n-1] = 1
                adjacency_matrix[self.n-1, i-1] = 1
        elif self.model == "linear":
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
        else:
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
                n_edges -= 1
            # If the density of the current adjacency matrix is over density, return it.
            if n_edges <= 0:
                return adjacency_matrix
            else:
                arr = [1]*n_edges + [0]*((self.n-1)*(self.n-2)//2 - n_edges)
                np.random.shuffle(arr)
                for i in range(0, self.n):
                    for j in range(i+2, self.n):
                        adjacency_matrix[i, j] = arr.pop()
                        adjacency_matrix[j, i] = adjacency_matrix[i, j]
        return adjacency_matrix
    
    def show_adjacency_matrix(self):
        print(self.adjacency_matrix)

    def show_graph_with_labels(self):
        rows, cols = np.where(self.adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos=pos, with_labels=True)
        plt.show()
