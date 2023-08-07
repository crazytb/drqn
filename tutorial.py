# https://www.youtube.com/@cartoonsondemand

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def stepfunc(thres, x):
    if x > thres:
        return 1
    else:
        return 0

class CustomEnv(gym.Env):
    def __init__(self, 
                 max_comp_units=100, 
                 max_terminals=10, 
                 max_epoch_size=100,
                 max_queue_size=10):
        super(CustomEnv, self).__init__()
        self.max_comp_units = max_comp_units
        self.max_terminals = max_terminals
        self.max_epoch_size = max_epoch_size
        self.max_queue_size = max_queue_size

        self.reward_weight = 0.5
        self.max_available_computation_units = max_comp_units
        self.max_number_of_associated_terminals = max_terminals
        self.max_channel_quality = 2
        self.max_remain_epochs = max_epoch_size
        # self.max_comp_units = np.array([max_comp_units] * max_queue_size)
        self.max_comp_units = np.array([10] * max_queue_size)
        self.max_proc_times = np.array([max_epoch_size] * max_queue_size)

        # 0: process, 1: offload
        self.action_space = spaces.Discrete(2)

        self.reward = 0

        self.observation_space = spaces.Dict({
            "available_computation_units": spaces.Discrete(self.max_available_computation_units),
            "number_of_associated_terminals": spaces.Discrete(self.max_number_of_associated_terminals),
            "channel_quality": spaces.Discrete(self.max_channel_quality),
            "remain_epochs": spaces.Discrete(self.max_remain_epochs),
            "mec_comp_units": spaces.MultiDiscrete([max_comp_units] * max_queue_size),
            "mec_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_queue_size),
            "queue_comp_units": spaces.MultiDiscrete([max_comp_units] * max_queue_size),
            "queue_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_queue_size),
        })
        self.rng = default_rng()
        self.current_obs = None
       
    def get_obs(self):
        return {"available_computation_units": self.available_computation_units,
                "number_of_associated_terminals": self.number_of_associated_terminals,
                "channel_quality": self.channel_quality,
                "remain_epochs": self.remain_epochs,
                "mec_comp_units": self.mec_comp_units,
                "mec_proc_times": self.mec_proc_times,
                "queue_comp_units": self.queue_comp_units,
                "queue_proc_times": self.queue_proc_times}
    
    def change_channel_quality(self):
        # State settings
        velocity = 10   # km/h
        snr_thr = 15
        snr_ave = snr_thr + 10
        f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
        speedoflight = 300000   # km/sec
        f_d = velocity/(3600*speedoflight)*f_0  # Hz
        packettime = 300    # us
        fdtp = f_d*packettime/1e6
        TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
        TRAN_00 = 1 - TRAN_01
        # TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_11 = 1 - TRAN_10

        if self.channel_quality == 0:  # Bad state
            if stepfunc(TRAN_00, random.random()) == 0: # 0 to 0
                channel_quality = 0
            else:   # 0 to 1
                channel_quality = 1
        else:   # Good state
            if stepfunc(TRAN_11, random.random()) == 0: # 1 to 1
                channel_quality = 1
            else:   # 1 to 0
                channel_quality = 0
    
        return channel_quality
    
    def fill_first_zero(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == 0:
                arr[i] = value
                break
        return arr

    def reset(self, seed=None, options=None):
        """
        Returns: observation
        """
        super().reset(seed=seed)

        self.available_computation_units = self.max_available_computation_units
        self.number_of_associated_terminals = self.rng.integers(1, self.max_number_of_associated_terminals + 1)
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.remain_epochs = self.max_remain_epochs
        self.remain_processing = 0
        self.mec_comp_units = np.zeros(self.max_queue_size, dtype=int)
        self.mec_proc_times = np.zeros(self.max_queue_size, dtype=int)
        self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)

        self.reward = 0

        observation = self.get_obs()
        
        return observation, {}
    
    def step(self, action):
        """
        Returns: observation, reward, terminated, truncated, info
        """

        # forwarding phase
        # 0: process, 1: offload
        if action == 0:
            # if available computation units are enough to process the first queue task and mec_comp_unit has empty slot and queue_comp_units has nonzero value.
            case_action = ((self.available_computation_units >= self.queue_comp_units[0]) and 
                           (self.mec_comp_units[self.mec_comp_units == 0].size > 0) and
                           (self.queue_comp_units[0] > 0))
            if case_action:
                self.available_computation_units -= self.queue_comp_units[0]
                self.mec_comp_units = self.fill_first_zero(self.mec_comp_units, self.queue_comp_units[0])
                self.mec_proc_times = self.fill_first_zero(self.mec_proc_times, self.queue_proc_times[0])
                self.queue_comp_units = np.concatenate([self.queue_comp_units[1:], np.array([0])])
                self.queue_proc_times = np.concatenate([self.queue_proc_times[1:], np.array([0])])
            else:
                self.reward += -100 # penalty

            self.channel_quality = self.change_channel_quality()
            self.remain_epochs = self.remain_epochs - 1
        else:
            self.reward += self.reward_weight if self.channel_quality == 1 else 0
            self.channel_quality = self.change_channel_quality()
            self.remain_epochs = self.remain_epochs - 1
            # shift left-wise queue information with 1 and pad 0
            self.queue_comp_units = np.concatenate([self.queue_comp_units[1:], np.array([0])])
            self.queue_proc_times = np.concatenate([self.queue_proc_times[1:], np.array([0])])

        # processing phase
        self.mec_proc_times = np.clip(self.mec_proc_times - 1, 0, self.max_proc_times)
        recovered_comp_units = self.mec_comp_units[self.mec_proc_times == 0].sum()
        if recovered_comp_units > 0:
            self.reward += 1
        self.available_computation_units = self.available_computation_units + recovered_comp_units
        self.mec_proc_times = np.concatenate([self.mec_proc_times[self.mec_proc_times > 0], np.zeros(self.max_queue_size - len(self.mec_proc_times[self.mec_proc_times > 0]), dtype=int)])
        self.mec_comp_units[self.mec_proc_times == 0] = 0
        self.mex_comp_units = np.concatenate([self.mec_comp_units[self.mec_proc_times > 0], np.zeros(self.max_queue_size - len(self.mec_proc_times[self.mec_proc_times > 0]), dtype=int)])

        next_obs = self.get_obs()

        return next_obs, self.reward, self.remain_epochs == 0, False, {}


    def render(self):
        """
        Returns: None
        """
        pass

    def close(self):
        """
        Returns: None
        """
        pass


env = CustomEnv()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
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
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
def flatten_dict_values(dictionary):
    flattened = np.array([])
    for v in list(dictionary.values()):
        if isinstance(v, np.ndarray):
            flattened = np.concatenate([flattened, v])
        else:
            flattened = np.concatenate([flattened, np.array([v])])
    return flattened
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
state = flatten_dict_values(state)
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

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
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if device == torch.device("cuda") or device == torch.device("mps"):
    num_episodes = 100
else:
    num_episodes = 50


for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(flatten_dict_values(state), dtype=torch.float32, device=device).unsqueeze(0)
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for epoch in range(100):
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(flatten_dict_values(observation), dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(reward)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()