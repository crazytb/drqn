# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master

import sys
import math
import collections
import random
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import numpy as np



if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class Policy(nn.Module):
    def __init__(self, state_space, hidden_space, action_space):
        super(Policy, self).__init__()
        
        self.state_space = state_space
        self.hidden_space = hidden_space
        self.action_space = action_space

        self.linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.linear2 = nn.Linear(self.hidden_space, self.hidden_space)
        self.linear3 = nn.Linear(self.hidden_space, self.action_space)
        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=2)
        return x
    
    # def forward(self, x, h, c):
    #     x = F.relu(self.linear1(x))
    #     x, (h, c) = self.linear2(x, (h, c))
    #     x = self.linear3(x)
    #     x = F.softmax(x, dim=2)
    #     return x, h, c
    
    # def init_hidden_state(self, training=None):
    #     # assert training is not None, "training step parameter should be determined"
    #     # if training is True:
    #     #     return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
    #     # else:
    #     return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])
    

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


class EpochBuffer:
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


class REINFORCEAgent:
    def __init__(self, state_space, hidden_space, action_space):
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.episode_memory = EpisodeMemory(random_update=False, max_epi_num=100, max_epi_len=600, batch_size=1, lookup_step=1)
        self.epoch_buffer = EpochBuffer()
        self.policy = Policy(state_space, hidden_space, action_space).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.policy.train()

    def sample_action(self, obs):
        probs = self.policy.forward(obs)
        action = torch.multinomial(probs.squeeze(0), 1)
        prob = probs.gather(2, action.unsqueeze(2))
        log_prob = prob.log().flatten()
        entropy = -(probs*probs.log()).sum()
        entropy = entropy.flatten()

        return action[0], log_prob, entropy
    
    # def sample_action(self, obs, h, c):
    #     probs, new_h, new_c = self.policy.forward(obs, h, c)
    #     action = torch.multinomial(probs.squeeze(0), 1)
    #     prob = probs.gather(2, action.unsqueeze(2))
    #     log_prob = prob.log()
    #     entropy = -(probs*probs.log()).sum()

    #     return action[0], new_h, new_c, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(R.expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.policy.parameters(), 40)
        self.optimizer.step()