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

class AdjacencyMatrix():
    def __init__(self, n: int, density: float):
        self.n = n
        self.density = density
        self.adjacency_matrix = np.zeros((self.n, self.n))
    
    def make_adjacency_matrix(self) -> np.ndarray:
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
                
        for i in range(1, self.n):
            self.adjacency_matrix[i-1, i] = 1
            self.adjacency_matrix[i, i-1] = 1
            n_edges -= 1
        
        # If the density of the current adjacency matrix is over density, return it.
        if n_edges > 0:
            arr = [1]*n_edges + [0]*((self.n-1)*(self.n-2)//2 - n_edges)
            np.random.shuffle(arr)
            for i in range(0, self.n):
                for j in range(i+2, self.n):
                    self.adjacency_matrix[i, j] = arr.pop()
                    self.adjacency_matrix[j, i] = self.adjacency_matrix[i, j]

    def print_adjacency_matrix(self):
        return self.adjacency_matrix

    def show_graph_with_labels(self):
        rows, cols = np.where(self.adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos=pos, with_labels=True)
        plt.show()
    

class Agent():
    def __init__(self):
        self.Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

    class ReplayMemory():
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)
        def push(self, *args):
            """Save a transition"""
            self.memory.append(self.Transition(*args))
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
        def __len__(self):
            return len(self.memory)

    def optimize_model(self, policy_net, target_net, optimizer):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
    def select_action(state, steps_done, policy_net, n_actions):
        sample = random.random()
        eps_threshold = 0.1
        if sample > eps_threshold:
            with torch.no_grad():
                # Return the action with the largest expected reward
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


    class DRQN(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(Agent.DRQN, self).__init__()
            
            self.hidden_space = 64
            self.state_space = n_observations
            self.action_space = n_actions
            
            self.layer1 = nn.Linear(self.state_space, self.hidden_space)
            self.layer2 = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
            self.layer3 = nn.Linear(self.hidden_space, self.action_space)
        
        def forward(self, x, h, c):
            x = F.relu(self.layer1(x))
            x, (new_h, new_c) = self.layer2(x, (h, c))
            x = F.softmax(self.layer3(x), dim=1)
            return x, new_h, new_c
    

class PNDEnv(Env):
    def __init__(self, n=10):
        super(PNDEnv, self).__init__()
        self.n = n
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
        transmission_prob = np.reshape(self._px_prob, newshape=(self.n))
        prev_result = np.reshape(self._prev_result, newshape=(self.n))
        return np.concatenate([transmission_prob, prev_result])
    
    def _get_info(self):
        print("Transmission Probability, Prev Result")
        for i in range(self.n):
            print(f"Node {i}: {self._px_prob[i]}, {self._prev_result[i]}")
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # State reset
        self._px_prob = np.random.uniform(0, 1, size=(self.n, 1))
        self._prev_result = np.zeros((self.n, 1))
        observation = self._get_obs()
        info = None
        return observation, info

    def step(self, accumulated_action, adjacency):  # 여기 해야 함.
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
