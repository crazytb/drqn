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

def make_adjacency_matrix(n: int, density: float=0.1, model: str=None) -> np.ndarray:
    """Make adjacency matrix of a clique network.
    
    Args:
        n (int): Number of nodes.
        density (float): Density of the clique network.
    
    Returns:
        np.ndarray: Adjacency matrix.
    """
    if density < 0 or density > 1:
        raise ValueError("Density must be between 0 and 1.")
    
    n_edges = int(n * (n - 1) / 2 * density)
    adjacency_matrix = np.zeros((n, n))

    if model == "dumbbell":
        adjacency_matrix[0, n-1] = 1
        adjacency_matrix[n-1, 0] = 1
        for i in range(1, n//2):
            adjacency_matrix[0, i] = 1
            adjacency_matrix[i, 0] = 1
        for i in range(n//2+1, n):
            adjacency_matrix[i-1, n-1] = 1
            adjacency_matrix[n-1, i-1] = 1
    elif model == "linear":
        for i in range(1, n):
            adjacency_matrix[i-1, i] = 1
            adjacency_matrix[i, i-1] = 1
    else:
        for i in range(1, n):
            adjacency_matrix[i-1, i] = 1
            adjacency_matrix[i, i-1] = 1
            n_edges -= 1
        # If the density of the current adjacency matrix is over density, return it.
        if n_edges <= 0:
            return adjacency_matrix
        else:
            arr = [1]*n_edges + [0]*((n-1)*(n-2)//2 - n_edges)
            np.random.shuffle(arr)
            for i in range(0, n):
                for j in range(i+2, n):
                    adjacency_matrix[i, j] = arr.pop()
                    adjacency_matrix[j, i] = adjacency_matrix[i, j]
    return adjacency_matrix

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
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

    class Q_net(nn.Module):
        def __init__(self, state_space=None,
                    action_space=None):
            super(self.Q_net, self).__init__()
            # space size check
            assert state_space is not None, "None state_space input: state_space should be selected."
            assert action_space is not None, "None action_space input: action_space should be selected."

            self.hidden_space = 64
            self.state_space = state_space
            self.action_space = action_space

            self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
            self.lstm    = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
            self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

        def forward(self, x, h, c):
            x = F.relu(self.Linear1(x))
            x, (new_h, new_c) = self.lstm(x,(h,c))
            x = self.Linear2(x)
            return x, new_h, new_c

        def sample_action(self, obs, h,c, epsilon):
            output = self.forward(obs, h,c)

            if random.random() < epsilon:
                return random.randint(0,1), output[1], output[2]
            else:
                return output[0].argmax().item(), output[1] , output[2]
        
        def init_hidden_state(self, batch_size, training=None):
            assert training is not None, "training step parameter should be dtermined"
            if training is True:
                return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
            else:
                return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

    class EpisodeMemory():
        """Episode memory for recurrent agent"""
        def __init__(self, random_update=False, 
                        max_epi_num=100, max_epi_len=500,
                        batch_size=1,
                        lookup_step=None):
            self.random_update = random_update # if False, sequential update
            self.max_epi_num = max_epi_num
            self.max_epi_len = max_epi_len
            self.batch_size = batch_size
            self.lookup_step = lookup_step

            if (random_update is False) and (self.batch_size > 1):
                sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

            self.memory = collections.deque(maxlen=self.max_epi_num)

        def put(self, episode):
            self.memory.append(episode)

        def sample(self):
            sampled_buffer = []

            ##################### RANDOM UPDATE ############################
            if self.random_update: # Random update
                sampled_episodes = random.sample(self.memory, self.batch_size)
                
                check_flag = True # check if every sample data to train is larger than batch size
                min_step = self.max_epi_len

                for episode in sampled_episodes:
                    min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

                for episode in sampled_episodes:
                    if min_step > self.lookup_step: # sample buffer with lookup_step size
                        idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                        sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                        sampled_buffer.append(sample)
                    else:
                        idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                        sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                        sampled_buffer.append(sample)

            ##################### SEQUENTIAL UPDATE ############################           
            else: # Sequential update
                idx = np.random.randint(0, len(self.memory))
                sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

            return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

        def __len__(self):
            return len(self.memory)


    class EpisodeBuffer:
        """A simple numpy replay buffer."""

        def __init__(self):
            self.obs = []
            self.action = []
            self.reward = []
            self.next_obs = []
            self.done = []

        def put(self, transition):
            self.obs.append(transition[0])
            self.action.append(transition[1])
            self.reward.append(transition[2])
            self.next_obs.append(transition[3])
            self.done.append(transition[4])

        def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
            obs = np.array(self.obs)
            action = np.array(self.action)
            reward = np.array(self.reward)
            next_obs = np.array(self.next_obs)
            done = np.array(self.done)

            if random_update is True:
                obs = obs[idx:idx+lookup_step]
                action = action[idx:idx+lookup_step]
                reward = reward[idx:idx+lookup_step]
                next_obs = next_obs[idx:idx+lookup_step]
                done = done[idx:idx+lookup_step]

            return dict(obs=obs,
                        acts=action,
                        rews=reward,
                        next_obs=next_obs,
                        done=done)

        def __len__(self) -> int:
            return len(self.obs)


    def train(q_net=None, target_q_net=None, episode_memory=None,
            device=None, 
            optimizer = None,
            batch_size=1,
            learning_rate=1e-3,
            gamma=0.99):

        assert device is not None, "None Device input: device should be selected."

        # Get batch from replay buffer
        samples, seq_len = episode_memory.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])
            dones.append(samples[i]["done"])

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(device)
        actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
        next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(device)
        dones = torch.FloatTensor(dones.reshape(batch_size,seq_len,-1)).to(device)

        h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)

        q_target, _, _ = target_q_net(next_observations, h_target.to(device), c_target.to(device))

        q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
        targets = rewards + gamma*q_target_max*dones


        h, c = q_net.init_hidden_state(batch_size=batch_size, training=True)
        q_out, _, _ = q_net(observations, h.to(device), c.to(device))
        q_a = q_out.gather(2, actions)

        # Multiply Importance Sampling weights to loss        
        loss = F.smooth_l1_loss(q_a, targets)
        
        # Update Network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def seed_torch(seed):
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    def save_model(model, path='default.pth'):
            torch.save(model.state_dict(), path)
    

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
