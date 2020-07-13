import gym
import numpy as np
import sys
import pickle
import logging

logging.basicConfig(level=logging.DEBUG, filename='logs/logs.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

err_file = open('logs/error.txt', 'w')
log_file = open('logs/logs.txt', 'w')
# sys.stdout = log_file
# sys.stderr = err_file

env = gym.make('Pong-v0', frameskip=(2, 3))
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
logging.info(f'Observation space: {env.observation_space}')

import torch.nn.functional as F
import torch.nn as nn
import torch

from skimage.transform import resize
from skimage.color import rgb2gray
# preprocess function
# resize obs to 110 x 110
def preprocess(img):
  re_img = resize(img, (110, 110))
  out = np.array(rgb2gray(re_img)).astype(np.float32)
  return out

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=8, padding=3, stride=4)  # output = 28 x 28
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2) # output = 13 x 13
    self.fc1 = nn.Linear(13*13*32, 256)
    self.fc2 = nn.Linear(256, 6)
  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = out.view(-1, 13*13*32)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
logging.info(f"Training on device {device}.")

episodes = int(1e4)
mem_size = int(1e6)
memory = [() for _ in range(mem_size)]
step_counter = -1
model = Net().to(device=device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9, eps=1e-02)
loss = nn.MSELoss()
BS = 32
epsilon = 1
epsilon_decay = 0.9999
epsilon_min = 0.1
warmup = 1000
discount = 0.95
skip_steps = 4

env = gym.make('Pong-v0')

rewards = []
for episode in range(episodes):
  env.reset()
  action = env.action_space.sample()

  tot_reward = 0
  reward = 0
  obs = []
  for _ in range(skip_steps):
    state, rew, done, info = env.step(action)
    reward += rew
    obs.append(preprocess(state))
  obs = np.array(obs)[None, ...]
  tot_reward += reward

  mem_upper = -1
  while not done:
    step_counter += 1
    mem_upper = step_counter if mem_upper < step_counter else mem_upper
    epsilon = epsilon*epsilon_decay if epsilon > epsilon_min else epsilon_min
    action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(model(torch.from_numpy(obs).to(device=device)).detach().cpu().numpy()[0])
    
    reward = 0
    obs_new = []
    for _ in range(skip_steps):
      state, rew, done, info = env.step(action)
      reward += rew
      obs_new.append(preprocess(state))
    obs_new = np.array(obs_new)[None, ...]
    tot_reward += reward

    memory[step_counter % mem_size] = (obs, action, reward, obs_new, done)
    obs = obs_new

    if step_counter > warmup:
      mini_batch = np.random.choice(np.array(memory)[:mem_upper], BS)
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
        dones.append(done_s)
      targets = model(torch.tensor(states).squeeze().to(device=device))
      targets_next = model(torch.tensor(states_next).squeeze().to(device=device)).detach().cpu().numpy()
      labels = np.array(rewards) + (np.array(done)>0) * discount * np.max(targets_next, axis=1)
      losses = loss(targets.gather(1, torch.tensor(actions).unsqueeze(1).to(device=device)).squeeze(), torch.from_numpy(labels.astype(np.float32)).to(device=device))

      optimizer.zero_grad()
      losses.backward()
      optimizer.step()
  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} with reward {tot_reward}')
    logging.info(f'finished episode {episode} with reward {tot_reward}')
    with open('cache/rewards.pkl', 'wb') as f:
      pickle.dump(rewards, f)
  if episode % 500 == 0:
    torch.save(model, f'models/model_{episode}.pt')
  rewards.append(tot_reward)
with open('rewards.pkl', 'wb') as f:
  pickle.dump(rewards, f)
env.close()