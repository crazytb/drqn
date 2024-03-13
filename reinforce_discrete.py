import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        
        self.state_space = state_space
        self.hidden_size = state_space
        self.action_space = action_space

        self.linear1 = nn.Linear(self.state_space, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.action_space)
        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x, h, c):
        x = F.relu(self.linear1(x))
        x, (h, c) = self.lstm(x, (h, c))
        x = self.linear2(x)
        x = F.softmax(x, dim=2)
        return x, h, c


class REINFORCEAgent:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # self.model.train()

    def select_action(self, obs, h, c):
        probs = self.model.forward(obs, h, c)
        action = probs.multinomial().data
        prob = probs[:, action[0,0]].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action[0], log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
	utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()