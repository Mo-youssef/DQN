import gym
import numpy as np
import sys
import pickle
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="tensorboard/run4/")  # tensorboard writer

from util import *
from DQN_model import *

logging.basicConfig(level=logging.DEBUG, filename='logs/logs.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

err_file = open('logs/error.txt', 'w')
log_file = open('logs/logs.txt', 'w')
# sys.stdout = log_file
# sys.stderr = err_file

env = gym.make("PongNoFrameskip-v4")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
logging.info(f'Observation space: {env.observation_space}')



device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
logging.info(f"Training on device {device}.")

steps = int(5e3)
episodes = int(1e4)
mem_size = int(1e5)
memory = np.empty(mem_size, dtype=object)
step_counter = -1
BS = 32
epsilon = 1
epsilon_min = 0.02
epsilon_decay = (epsilon-epsilon_min)/1e5
warmup = 10000
discount = 0.95
skip_steps = 4

behavior_model = Net().to(device=device)
target_model = Net().to(device=device)
for param in target_model.parameters():
    param.requires_grad = False
model_copy = 500  # K
# when using RMSprop use (alpha=0.9, eps=1e-02)
optimizer = torch.optim.Adam(behavior_model.parameters(), lr=1e-4)
# loss = nn.MSELoss()
loss = nn.SmoothL1Loss()

rewards = []
for episode in range(episodes):
  env.reset()
  action = env.action_space.sample()

  tot_reward = 0
  obs, reward, done = skip_action(action, env, skip_steps)
#   obs = torch.as_tensor(obs, device=device)
  tot_reward += reward

  while not done:

    step_counter += 1
    if step_counter % model_copy == 0:
      target_model.load_state_dict(behavior_model.state_dict())

    epsilon -= epsilon_decay
    epsilon = max(epsilon, epsilon_min)
    action = env.action_space.sample() if np.random.rand() < epsilon else behavior_model(
        torch.as_tensor(obs, device=device)).argmax().item()

    obs_new, reward, done = skip_action(action, env, skip_steps)
    tot_reward += reward

    # obs_new = torch.as_tensor(obs_new, device=device)

    memory[step_counter % mem_size] = (obs, action, reward, obs_new, done)
    obs = obs_new

    if step_counter > warmup:
      mini_batch = np.random.choice(memory[:step_counter], BS)
      labels = []
      train_data = []
      states = []
      states_next = []
      actions = []
      rewards = []
      dones = []
      for (obs_s, action_s, reward_s, obs_new_s, done_s) in mini_batch:
        states.append(obs_s)
        states_next.append(obs_new_s)
        actions.append(action_s)
        rewards.append(reward_s)
        dones.append(done_s < 0.1)
      dones_tensor = torch.tensor(dones, device=device)
      rewards_tensor = torch.tensor(rewards, device=device)
      targets = behavior_model(torch.tensor(states, device=device).squeeze())
      with torch.no_grad():
        targets_next = target_model(
            torch.tensor(states_next, device=device).squeeze()).detach()
      labels = rewards_tensor + dones_tensor * discount * targets_next.max(dim=1)[0]
      losses = loss(targets.gather(1, torch.tensor(actions, device=device).unsqueeze(1)).squeeze(), labels)

      optimizer.zero_grad()
      losses.backward()
      optimizer.step()
  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')
    logging.info(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')
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
