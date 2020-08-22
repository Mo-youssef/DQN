import gym
import numpy as np
import sys
import pickle
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="tensorboard/run1/")  # tensorboard writer

from util import *
from DQN_model import *
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# logging.basicConfig(level=logging.DEBUG, filename='logs/logs.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# err_file = open('logs/error.txt', 'w')
# log_file = open('logs/logs.txt', 'w')
# sys.stdout = log_file
# sys.stderr = err_file

env = gym.make("PongNoFrameskip-v4")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
# logging.info(f'Observation space: {env.observation_space}')



device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
# logging.info(f"Training on device {device}.")

steps = int(5e3)
episodes = int(1e4)
step_counter = -1
BS = 32
epsilon = 1
epsilon_min = 0.02
epsilon_decay = (epsilon-epsilon_min)/1e5
warmup = 10000
discount = 0.95
skip_steps = 4

use_per = 1
use_gpu = 1
mem_size = int(1e5)
alpha_per = 0.6
beta_per = 0.7
memory = PrioritizedReplayBuffer(mem_size, alpha_per, use_gpu=use_gpu) if use_per else ReplayBuffer(mem_size, use_gpu=use_gpu)

double_dqn = 1
dueling = 1
grad_norm = 10
behavior_model = Net().to(device=device)
target_model = Net().to(device=device)
for param in target_model.parameters():
    param.requires_grad = False
model_copy = 1000  # K
# when using RMSprop use (alpha=0.9, eps=1e-02)
optimizer = torch.optim.Adam(behavior_model.parameters(), lr=1e-4)
# loss = nn.MSELoss()
loss = nn.SmoothL1Loss(reduction='none')

rewards = []
for episode in range(episodes):
  env.reset()
  action = env.action_space.sample()

  tot_reward = 0
  obs, reward, done = skip_action(action, env, skip_steps)
  obs = torch.as_tensor(obs, device=device) if use_gpu else obs
  tot_reward += reward

  while not done:

    step_counter += 1
    if step_counter % model_copy == 0:
      target_model.load_state_dict(behavior_model.state_dict())

    epsilon -= epsilon_decay
    epsilon = max(epsilon, epsilon_min)
    action = env.action_space.sample() if np.random.rand() < epsilon else behavior_model(
        obs).argmax().item() if use_gpu else behavior_model(
        torch.as_tensor(obs, device=device)).argmax().item()

    obs_new, reward, done = skip_action(action, env, skip_steps)
    tot_reward += reward

    obs_new = torch.as_tensor(obs_new, device=device) if use_gpu else obs_new

    memory.add(obs, action, reward, obs_new, done)
    obs = obs_new

    if step_counter > warmup:
      if use_per:
        states, actions, rewards, states_next, dones, weights, idxes = memory.sample(
            BS, beta_per)
        # states, actions, rewards, states_next, dones = data
      else:
        states, actions, rewards, states_next, dones = memory.sample(BS)

      states = torch.stack(states).squeeze() if use_gpu else torch.tensor(
          states, device=device).squeeze()
      states_next = torch.stack(states_next).squeeze() if use_gpu else torch.tensor(
          states_next, device=device).squeeze()
      actions_tensor = torch.tensor(actions, device=device)
      dones_tensor = torch.tensor(dones, device=device)
      rewards_tensor = torch.tensor(rewards, device=device)
      targets = behavior_model(states)
      with torch.no_grad():
        if double_dqn:
          targets_next = target_model(states_next).detach()
          double_targets = behavior_model(states_next).detach()
          q_value = targets_next.gather(1, double_targets.argmax(1).unsqueeze(1)).squeeze()
        else:
          targets_next = target_model(states_next).detach()
          q_value = targets_next.max(dim=1)[0]
      labels = rewards_tensor + (dones_tensor.logical_not_()) * discount * q_value
      predictions = targets.gather(1, actions_tensor.unsqueeze(1)).squeeze()
      losses = loss(predictions, labels)
      if use_per:
        losses.register_hook(lambda x: x * torch.as_tensor(weights, device=device, dtype=torch.float32))
      final_loss = losses.mean()
      # weighted_losses = losses * torch.as_tensor(weights, device=device) if use_per else losses
      # final_loss = weighted_losses.mean()
      optimizer.zero_grad()
      final_loss.backward()
      if dueling:
        torch.nn.utils.clip_grad_norm_(behavior_model.parameters(), grad_norm)
      optimizer.step()

      if use_per: 
        td_error = (labels - predictions).detach().cpu().numpy()
        memory.update_priorities(idxes, np.abs(td_error))
      

  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')
    # logging.info(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')
    with open('cache/rewards.pkl', 'wb') as f:
      pickle.dump(rewards, f)
  if episode % 200 == 0:
    torch.save(behavior_model, f'models/model_{episode}.pt')
  rewards.append(tot_reward)
  writer.add_scalar("Reward/Episode", tot_reward, episode)
  writer.add_scalar("Reward/Timestep", tot_reward, step_counter)
  writer.add_scalar("Epsilon/Episode", epsilon, episode)
  writer.add_scalar("Epsilon/Timestep", epsilon, step_counter)
with open('rewards.pkl', 'wb') as f:
  pickle.dump(rewards, f)
env.close()
writer.flush()
