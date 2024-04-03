# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# https://github.com/chingyaoc/pytorch-REINFORCE/tree/master

# % tensorboard --logdir=runs
# 이거 extreme edge device로 키워드 잡으면 좋을듯

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from environment import *
from reinforce_discrete import *


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Set parameters
batch_size = 1
learning_rate = 1e-3
min_epi_num = 20 # Start moment to train the Q network
episodes = 1000
target_update_period = 10
eps_start = 0.1
eps_end = 0.001
eps_decay = 0.995
tau = 1e-2

# DRQN param
random_update = False    # If you want to do random update instead of sequential update
lookup_step = 20        # If you want to do random update instead of sequential update

# Number of envs param
n_agents = 10
density = 1
max_step = 300
# None, "dumbbell", "linear", "fullmesh"
model = "fullmesh"


# Set gym environment
env_params = {
    "n": n_agents,
    "density": density,
    "max_episode_length": max_step,
    "model": model,
    }
if model == None:
    env_params_str = f"n{n_agents}_density{density}_max_episode_length{episodes}"
else:
    env_params_str = f"n{n_agents}_model{model}_max_episode_length{episodes}"

# Make env and reset it
env = PNDEnv(**env_params)
env.reset()

# Set the seed
# seed = 42
# np.random.seed(seed)
# random.seed(seed)
# seed_torch(seed)

# default `log_dir` is "runs" - we'll be more specific here
# run_path = 'runs/DRQN_'+env_params_str
output_path = 'outputs/DRQN_'+env_params_str
writer = SummaryWriter(filename_suffix=env_params_str)
if not os.path.exists(output_path):
    os.makedirs(output_path)
env.save_graph_with_labels(output_path)


# Create policy functions
n_states = len(env.observation_space)
n_hiddens = 32
n_actions = 2
agents = [REINFORCEAgent(n_states, n_hiddens, n_actions) for _ in range(n_agents)]

# Create dataframe to store the log
df = pd.DataFrame(columns
                  = ['episode', 'time']
                  + [f'action_{i}' for i in range(n_agents)]
                  + [f'age_{i}' for i in range(n_agents)])
appended_df = []

# Train
for i_epi in tqdm(range(episodes), desc="Episodes", position=0, leave=True):
    s, _ = env.reset()
    entropy_cum = [[] for _ in range(n_agents)]
    prob_cum = [[] for _ in range(n_agents)]
    reward_cum = []
    obs_cum = [s[x + n_agents*np.array(range(n_states))] for x in range(n_agents)]
    done = False

    h_cum, c_cum = zip(*[agents[i].policy.init_hidden_state() for i in range(n_agents)])
    
    for t in tqdm(range(max_step), desc="   Steps", position=1, leave=False):
        # Get action
        actions, h_cum, c_cum, probs, entropies = zip(*[agents[i].sample_action(torch.from_numpy(obs_cum[i]).float().to(device).unsqueeze(0).unsqueeze(0),
                                                                  h_cum[i].to(device),
                                                                  c_cum[i].to(device)) for i in range(n_agents)])
        a = np.array([a[0].item() for a in actions])

        for i in range(n_agents):
            prob_cum[i].append(probs[i])
            entropy_cum[i].append(entropies[i])        
        # log_prob_cum[:, t] = torch.stack(log_probs).flatten()
        # entropy_cum[:, t] = torch.stack(entropies).flatten()
        
        # Do action
        s_prime, r, done, _, _ = env.step(a)
        # Put r into reward_cum
        reward_cum.append(r)
        obs_prime_cum = [s_prime[x + n_agents*np.array(range(n_states))] for x in range(n_agents)]

        # make data
        done_mask = 0.0 if done else 1.0

        for i_n in range(n_agents):
            agents[i_n].epoch_buffer.put([obs_cum[i_n], a[i_n], r, obs_prime_cum[i_n], done_mask])

        obs_cum = obs_prime_cum

        df_currepoch = pd.DataFrame(data=[[i_epi, t, *a, *env.get_current_age()]],
                                    columns=['episode', 'time'] + [f'action_{i}' for i in range(n_agents)] + [f'age_{i}' for i in range(n_agents)])
        appended_df.append(df_currepoch)
        
        if done:
            break
        
    for i_m in range(n_agents):
        agents[i_m].update_parameters(reward_cum, prob_cum[i_m], entropy_cum[i_m], 0.98)
        # train_policy(agents[i_m].policy, agents[i_m].episode_memory, device, optimizer=agents[i_m].optimizer, batch_size=batch_size, learning_rate=learning_rate)

        # if (t+1) % target_update_period == 0:
        #     for target_param, local_param in zip(Q_target_cum[i_m].parameters(), Policy_cum[i_m].parameters()): # <- soft update
        #             target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
    
    # for i_j in range(n_agents):
    #     agents[i_j].episode_memory.put(agents[i_j].epoch_buffer)
    
    # epsilon = max(eps_end, epsilon * eps_decay) # Linear annealing
    prob_list = []
    for prob_tensor in probs:
        prob_list.append(round(prob_tensor[-1].item(), 2))
    
    print(f"n_episode: {i_epi}/{episodes}, score: {reward_cum[-1]:.2f}, probs: {prob_list}")
    
    # Log the reward
    writer.add_scalar('Rewards per episodes', reward_cum[-1], i_epi)
    
for i in range(n_agents):
    torch.save(agents[i].policy.state_dict(), output_path + f"/Q_cum_{i}.pth")
    
df = pd.concat(appended_df, ignore_index=True)

# Save the log with timestamp
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
df_first10episodes = df[df['episode'] <= 10]
df_last10episodes = df[df['episode'] >= episodes-10]
df_tobestored = pd.concat([df_first10episodes, df_last10episodes], ignore_index=True)
df_tobestored.to_csv(output_path + f"/log_{current_time}.csv", index=False)

writer.close()
env.close()

print(env_params_str)