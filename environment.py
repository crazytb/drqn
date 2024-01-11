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
BUFFERSIZE = 20  # Def. 20
NUMNODES = 10
DIMSTATES = 2 * NUMNODES + 1
FRAMETIME = 270  # microseconds
TIMEEPOCH = 300  # microseconds
FRAMETXSLOT = 30
FRAMEAGGLIMIT = int(TIMEEPOCH/FRAMETXSLOT)
BEACONINTERVAL = 100_000  # microseconds
# MAXAOI = int(np.ceil(BEACONINTERVAL / TIMEEPOCH))
ACCESSPROB = 1 / NUMNODES
# ACCESSPROB = 1
POWERCOEFF = 0.1
AOIPENALTY = 1
PER = 0.1
PEAKAOITHRES = 20_000   # That is, 5 000 for 5ms, (5,20)


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
        return self.layer3(x)


class OneHotEncoding(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        one_hot_vector = np.zeros(self.size)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).contains(0)
            number_of_ones = list(x).contains(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size


class PNDEnv(Env):
    def __init__(self):
        super(PNDEnv, self).__init__()
        # Actions we can take 0) transmit and 1) listen
        self.action_space = Discrete(2)
        # Observation space
        self.observation_space = spaces.Dict({
            "transmission_prob": spaces.Box(low=0, high=1, shape=(1, 1)),
            # [1, 0, 0, 0]: Transmission,
            # [0, 1, 0, 0]: Listen and No Collision,
            # [0, 0, 1, 0]: Listen and Collision Detected,
            # [0, 0, 0, 1]: Listen and Channel Idle,
            "prev_result": OneHotEncoding(size=4),
        })
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # State reset
        self.txprob = np.random.uniform(0, 1)
        self.prev_result = np.array([1, 0, 0])
        self.current_aois = np.zeros([NUMNODES], dtype=float)
        self.node_location = BUFFERSIZE * np.ones([NUMNODES], dtype=int)
        self.node_aoi = np.zeros([NUMNODES], dtype=float)
        
        self.leftslots = round(BEACONINTERVAL / TIMEEPOCH)
        self.leftbuffers = BUFFERSIZE
        self.current_time = 0
        self.consumed_energy = 0
        self.inbuffer_info_node = np.zeros([BUFFERSIZE], dtype=int)
        self.inbuffer_info_timestamp = np.zeros([BUFFERSIZE], dtype=int)
        self.insert_index = 0
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        return self.current_obs, self.info


    def probenqueue(self, dflog):
        self.current_time += TIMEEPOCH / BEACONINTERVAL
        self.current_aois += TIMEEPOCH / BEACONINTERVAL
        
        # Define condition that the elements of the dflog can enqueue.
        cond = ((dflog.time >= self.current_time*BEACONINTERVAL - TIMEEPOCH) 
                & (dflog.time < self.current_time*BEACONINTERVAL))
        
        # Extract target dflog
        targetdflog = dflog[cond][:self.leftbuffers]
        tnodenumber = min(len(targetdflog), self.leftbuffers)
        self.leftbuffers -= tnodenumber

        if tnodenumber == 0:
            pass
        else:
            enquenode = targetdflog.node.values.astype(int)
            enquenodetimestamp = targetdflog.timestamp.values.astype(int)

            self.inbuffer_info_node[self.insert_index:self.insert_index + tnodenumber] = enquenode
            self.inbuffer_info_timestamp[self.insert_index:self.insert_index + tnodenumber] = enquenodetimestamp
            self.insert_index += tnodenumber

            self.node_location, self.node_aoi = self._get_node_info(self.inbuffer_info_node, self.inbuffer_info_timestamp)
            
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

    def step(self, action):  # 여기 해야 함.
        reward = 0
        # 0: FORWARD
        if action == 0:
            if self._is_buffer_empty():
                pass
            else:
                dequenodes = self.inbuffer_info_node[self.inbuffer_info_node != 0][:FRAMEAGGLIMIT]
                dequenodeaoi_timestamps = self.inbuffer_info_timestamp[self.inbuffer_info_node != 0][:FRAMEAGGLIMIT]
                num_dequenodes = len(dequenodes)
                
                if self.channel_quality == 0:
                    for dequenode, dequenodeaoi_timestamp in zip(dequenodes, dequenodeaoi_timestamps):
                        self.current_aois[dequenode - 1] = self.current_time - (dequenodeaoi_timestamp/BEACONINTERVAL)
                
                # Left-shift bufferinfo
                self.inbuffer_info_node[:-num_dequenodes] = self.inbuffer_info_node[num_dequenodes:]
                self.inbuffer_info_node[-num_dequenodes:] = 0
                self.inbuffer_info_timestamp[:-num_dequenodes] = self.inbuffer_info_timestamp[num_dequenodes:]
                self.inbuffer_info_timestamp[-num_dequenodes:] = 0
                self.leftbuffers += num_dequenodes
                self.insert_index -= num_dequenodes
            reward -= POWERCOEFF*0.308
            self.consumed_energy += 280 * 1.1 * FRAMETIME    # milliamperes * voltage * time

        # 1: Flush
        elif action == 1:
            self.inbuffer_info_node.fill(0)
            self.inbuffer_info_timestamp.fill(0)
            self.leftbuffers = BUFFERSIZE
            self.insert_index = 0

        # 2: Leave
        elif action == 2:
            pass
        
        self.node_location, self.node_aoi = self._get_node_info(self.inbuffer_info_node, self.inbuffer_info_timestamp)
        self.channel_quality = self._change_channel_quality()
        self.info = self._get_obs()
        self.current_obs = self._flatten_dict_values(self.info)

        self.leftslots -= 1
        done = self.leftslots <= 0
        
        # if self.current_aois.max() >= (PEAKAOITHRES / BEACONINTERVAL):
        reward -= np.clip(self.current_aois - (PEAKAOITHRES / BEACONINTERVAL), 0, None).sum()
        # count the number of nodes whose aoi is less than PEAKAOITHRES / BEACONINTERVAL
        # reward += np.count_nonzero(self.current_aois < (PEAKAOITHRES / BEACONINTERVAL)) * (1/NUMNODES)
        
        return self.current_obs, reward, False, done, self.info

    
    def render(self):
        # Implement viz
        pass
