import gym
import numpy as np
import sys
import pickle
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pfrl
from pfrl import agents
from pfrl import experiments
from pfrl import explorers
from pfrl import utils
from pfrl.q_functions import DuelingDQN
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers

tensorboard_dir_name = 'PER_DOUBLE_DUELING_pfrl'
tensorboard_dir = f"tensorboard/{tensorboard_dir_name}"
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(log_dir=tensorboard_dir)

from DQN_model import Net


env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(
    "PongNoFrameskip-v4", max_frames=30 * 60 * 60), episode_life=True, clip_rewards=True)

double_dqn = 1
dueling = 1
grad_norm = 10
q_func = Net(actions=env.action_space.n, dueling=dueling)

model_copy = 1000  # K
optimizer = torch.optim.Adam(q_func.parameters(), lr=1e-4)
loss = nn.SmoothL1Loss(reduction='none')



rbuf = replay_buffers.PrioritizedReplayBuffer(
    10 ** 5,
    alpha=0.6,
    beta0=0.4,
    betasteps=1e7,
    num_steps=1,
)


def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255


episodes = int(1e4)
step_counter = -1
BS = 32
epsilon_min = 0.02
warmup = 10000
discount = 0.95
skip_steps = 4

explorer = explorers.LinearDecayEpsilonGreedy(
    1.0,
    epsilon_min,
    1e5,
    lambda: np.random.randint(env.action_space.n),
)

agent = agents.DoubleDQN(
    q_func,
    optimizer,
    rbuf,
    gpu=0,
    gamma=discount,
    explorer=explorer,
    replay_start_size=warmup,
    target_update_interval=model_copy,
    clip_delta=True,
    update_interval=1,
    batch_accumulator="mean",
    phi=phi,
)





for episode in range(episodes):
  obs = env.reset()
  tot_reward = 0
  tstep = 0
  done = 0
  while not done:
    tstep += 1
    step_counter += 1
    action = agent.act(obs)

    obs, reward, done, _ = env.step(action)
    tot_reward += reward
    reset = tstep == 30*60*60
    agent.observe(obs, reward, done, reset)
    if done or reset:
        break

  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')

  writer.add_scalar("Reward/Episode", tot_reward, episode)
  writer.add_scalar("Reward/Timestep", tot_reward, step_counter)

env.close()
writer.flush()
