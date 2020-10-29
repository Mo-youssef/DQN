import gym
import numpy as np
import sys
import pickle
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

tensorboard_dir_name = 'PER_DOUBLE_DUELING_nogpu'
tensorboard_dir = f"tensorboard/{tensorboard_dir_name}"
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(log_dir=tensorboard_dir)

from util import preprocess, skip_action
from DQN_model import Net
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

env = gym.make("MontezumaRevengeNoFrameskip-v4")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)



device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

steps = int(5e3)
episodes = int(1e5)
step_counter = -1
BS = 32
warmup = 10000
discount = 0.95
skip_steps = 4
train_every = 4
train_steps = -1
epsilon = 1
epsilon_min = 0.02
epsilon_decay = (epsilon-epsilon_min)/1e6
epsilon_method = 1
epsilon_range = np.linspace(0.02, 0.7, 256)

use_per = 1
use_gpu = 0
mem_size = int(1e6)
alpha_per = 0.7
beta_per = 0.5
beta_per_rate = (1-beta_per)/1e7
memory = PrioritizedReplayBuffer(mem_size, alpha_per, use_gpu=use_gpu) if use_per else ReplayBuffer(mem_size, use_gpu=use_gpu)

double_dqn = 1
dueling = 1
grad_norm = 10
behavior_model = Net(actions=env.action_space.n, dueling=dueling).to(device=device)
target_model = Net(actions=env.action_space.n, dueling=dueling).to(device=device)
for param in target_model.parameters():
    param.requires_grad = False
model_copy = 1500  # K
optimizer = torch.optim.Adam(behavior_model.parameters(), lr=1e-4)
loss = nn.SmoothL1Loss(reduction='none')

for episode in range(episodes):
  env.reset()
  action = env.action_space.sample()

  tot_reward = 0
  obs, reward, done = skip_action(action, env, skip_steps)
  obs = torch.as_tensor(obs, device=device, dtype=torch.float32) if use_gpu else obs
  tot_reward += reward

  while not done:

    step_counter += 1
    if step_counter % model_copy == 0:
      target_model.load_state_dict(behavior_model.state_dict())

    if epsilon_method == 0:
      epsilon -= epsilon_decay
      epsilon = max(epsilon, epsilon_min)
    else:
      epsilon = np.random.choice(epsilon_range)

    action = env.action_space.sample() if np.random.rand() < epsilon else behavior_model(
        obs).argmax().item() if use_gpu else behavior_model(
        torch.as_tensor(obs, device=device)).argmax().item()

    obs_new, reward, done = skip_action(action, env, skip_steps)
    tot_reward += reward

    obs_new = torch.as_tensor(obs_new, device=device, dtype=torch.float32) if use_gpu else obs_new

    memory.add(obs, action, reward, obs_new, done)
    obs = obs_new

    if (step_counter > warmup) and (step_counter % train_every == 0):
      train_steps += 1
      beta_per += beta_per_rate
      if use_per:
        states, actions, rewards, states_next, dones, weights, idxes = memory.sample(
            BS, beta_per)
      else:
        states, actions, rewards, states_next, dones = memory.sample(BS)

      states = torch.stack(states).squeeze() if use_gpu else torch.tensor(
          states, device=device, dtype=torch.float32).squeeze()
      states_next = torch.stack(states_next).squeeze() if use_gpu else torch.tensor(
          states_next, device=device, dtype=torch.float32).squeeze()
      actions_tensor = torch.tensor(actions, device=device)
      dones_tensor = torch.tensor(dones, device=device)
      rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
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
        memory.update_priorities(idxes, np.abs(td_error+0.001*np.min(td_error[td_error!=0])))
      

  if episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')

  if episode % 200 == 0:
    torch.save(behavior_model, f'models/model_{episode}.pt')
  writer.add_scalar("Reward/Episode", tot_reward, episode)
  writer.add_scalar("Reward/Timestep", tot_reward, step_counter)
  writer.add_scalar("Reward/Trainstep", tot_reward, train_steps)
  writer.add_scalar("Epsilon/Episode", epsilon, episode)
  writer.add_scalar("Epsilon/Timestep", epsilon, step_counter)
  writer.add_scalar("Buffer Size/Trainstep", len(memory), train_steps)

env.close()
writer.flush()
