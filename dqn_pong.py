import gym
import numpy as np
import sys
import pickle
import logging
from skimage.transform import resize
from skimage.color import rgb2gray
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="tensorboard/run2/")  # tensorboard writer

logging.basicConfig(level=logging.DEBUG, filename='logs/logs.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

err_file = open('logs/error.txt', 'w')
log_file = open('logs/logs.txt', 'w')
# sys.stdout = log_file
# sys.stderr = err_file

env = gym.make("PongNoFrameskip-v4")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
logging.info(f'Observation space: {env.observation_space}')

def preprocess(img):
  re_img = resize(img, (84, 84))
  out = np.array(rgb2gray(re_img)).astype(np.float32)
  return out

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(4, 16, kernel_size=8, padding=3, stride=4)  # output = 28 x 28
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=0, stride=2) # output = 13 x 13
    self.fc1 = nn.Linear(2592, 256)
    self.fc2 = nn.Linear(256, 6)
  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = out.view(-1, 2592)
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    return out

def skip_action(action):
    reward = 0
    obs = []
    for _ in range(skip_steps):
      state, rew, done, _ = env.step(action)
      reward += rew
      obs.append(preprocess(state))
    obs = np.array(obs)[None, ...]
    return obs, reward, done

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")
logging.info(f"Training on device {device}.")

steps = int(5e3)
episodes = int(1e4)
mem_size = int(1e5)
memory = np.empty(mem_size, dtype=object)
step_counter = -1
behavior_model = Net().to(device=device)
target_model = Net().to(device=device)
for param in target_model.parameters():
    param.requires_grad = False
model_copy = 500  # K
optimizer = torch.optim.Adam(behavior_model.parameters(), lr=1e-4)  # when using RMSprop use (alpha=0.9, eps=1e-02)
loss = nn.MSELoss()

BS = 32
epsilon = 1
epsilon_min = 0.02
epsilon_decay = (epsilon-epsilon_min)/1e6
warmup = 10000
discount = 0.95
skip_steps = 4

env = gym.make("PongNoFrameskip-v4")

rewards = []
for episode in range(episodes):
  env.reset()
  action = env.action_space.sample()

  tot_reward = 0
  obs, reward, done = skip_action(action)
  tot_reward += reward

  for step in range(steps):
    if done:
      break

    step_counter += 1
    if step_counter % model_copy == 0:
      target_model.load_state_dict(behavior_model.state_dict())

    epsilon -= epsilon_decay
    action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(
        behavior_model(torch.from_numpy(obs).to(device=device)).detach().cpu().numpy()[0])

    obs_new, reward, done = skip_action(action)
    tot_reward += reward

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
      targets = behavior_model(torch.tensor(states).squeeze().to(device=device))
      with torch.no_grad():
        targets_next = target_model(torch.tensor(states_next).squeeze().to(device=device)).detach()
      # labels = np.array(rewards) + (np.array(dones)>0) * discount * np.max(targets_next, axis=1)
      labels = rewards_tensor + dones_tensor * discount * targets_next.max(dim=1)[0]
      # losses = loss(targets.gather(1, torch.tensor(actions).unsqueeze(1).to(device=device)).squeeze(), torch.from_numpy(labels.astype(np.float32)).to(device=device))
      losses = loss(targets.gather(1, torch.tensor(actions, device=device).unsqueeze(1)).squeeze(), labels)

      optimizer.zero_grad()
      losses.backward()
      optimizer.step()
  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')
    logging.info(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')
    with open('cache/rewards.pkl', 'wb') as f:
      pickle.dump(rewards, f)
  if episode % 500 == 0:
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
