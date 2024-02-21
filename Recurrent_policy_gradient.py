import sys
import os
from typing import Dict, List, Tuple
import gymnasium as gym
import collections
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from environment_rev import PNDEnv

# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
    
class Policy(nn.Module):
    def __init__(self, state_space=2, action_space=2):
        super(Policy, self).__init__()
        self.data = []
        self.state_space = state_space
        self.hidden_space = 4
        self.action_space = action_space
        self.linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space)
        self.linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = F.softmax(self.linear2(x), dim=2)
        return x, new_h, new_c

    def put_data(self, transition):
        self.data.append(transition)

    def sample_action(self, obs, h, c):
        output = self.forward(obs, h, c)
        # Select action with respect to the action probabilities
        action = torch.squeeze(output[0]).multinomial(num_samples=1)    
        return action.item(), output[1], output[2]
    
    def init_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_space, device=device), torch.zeros(1, 1, self.hidden_space, device=device)

def train(pi, optimizer):
    R = 0
    policy_loss = []
    optimizer.zero_grad()
    for r, prob in pi.data[::-1]:
        R = r + gamma * R
        loss = -torch.log(prob) * R # Negative score function x reward
        policy_loss.append(loss)
    sum(policy_loss).backward()
    optimizer.step()
    pi.data = []
    

# Set parameters
batch_size = 8
learning_rate = 1e-3
buffer_len = int(100000)
min_epi_num = 10 # Start moment to train the Q network
episodes = 500
target_update_period = 10
eps_start = 0.1
eps_end = 0.001
eps_decay = 0.995
tau = 1e-2

# DRQN param
random_update = True    # If you want to do random update instead of sequential update
lookup_step = 20        # If you want to do random update instead of sequential update

# Number of envs param
n_nodes = 10
n_agents = 10
density = 1
max_step = 300
model = None

# Set gym environment
env_params = {
    "n": n_nodes,
    "density": density,
    "max_episode_length": max_step,
    "model": model
    }
if model == None:
    env_params_str = f"n{n_nodes}_density{density}_max_episode_length{max_step}"
else:
    env_params_str = f"n{n_nodes}_model{model}_max_episode_length{max_step}"
    
env = PNDEnv(**env_params)
env.reset()

output_path = 'outputs/DRQN_'+env_params_str
writer = SummaryWriter(filename_suffix=env_params_str)
if not os.path.exists(output_path):
    os.makedirs(output_path)
env.save_graph_with_labels(output_path)

# Create Policy functions
n_states = 2
n_actions = 2

pi_cum = [Policy(state_space=n_states, action_space=n_actions).to(device) for _ in range(n_agents)]

# Set optimizer
score = 0
score_sum = 0
optimizer_cum = [optim.Adam(pi_cum[i].parameters(), lr=learning_rate) for i in range(n_agents)]

epsilon = eps_start

df = pd.DataFrame(columns=['episode', 'time'] + [f'action_{i}' for i in range(n_agents)] + [f'age_{i}' for i in range(n_agents)])
appended_df = []

for i_epi in tqdm(range(episodes), desc="Episodes", position=0, leave=True):
    s, _ = env.reset()
    obs_cum = [s[np.array([x, x+n_agents])] for x in range(n_agents)]
    h_cum, c_cum = zip(*[pi_cum[i].init_hidden_state() for i in range(n_agents)])
    done = False
    
    for t in tqdm(range(max_step), desc="   Steps", position=1, leave=False):
        prob_cum = [pi_cum[i](torch.from_numpy(obs_cum[i]).float().unsqueeze(0).unsqueeze(0).to(device), h_cum[i].to(device), c_cum[i].to(device))[0] for i in range(n_agents)]
        a_cum, h_cum, c_cum = zip(*[pi_cum[i].sample_action(torch.from_numpy(obs_cum[i]).float().unsqueeze(0).unsqueeze(0).to(device), h_cum[i].to(device), c_cum[i].to(device)) for i in range(n_agents)])
        a_cum = np.array(a_cum)
        s_prime, r, done, _, info = env.step(a_cum)
        done_mask = 0.0 if done else 1.0
        for i in range(n_agents):
            a = a_cum[i]
            pi_cum[i].put_data((r, prob_cum[i][0, 0, a]))
        obs_cum = [s_prime[np.array([x, x+n_agents])] for x in range(n_agents)]
        score += r
        
        if done:
            break

    for pi, optimizer in zip(pi_cum, optimizer_cum):
        train(pi, optimizer)
    
    print(f"n_episode: {i_epi}/{episodes}, score: {score}")
    writer.add_scalar('Rewards per episodes', score, i_epi)
    score = 0

env.close()