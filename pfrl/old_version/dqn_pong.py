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
from double_dqn import DoubleDQN
from pfrl import experiments
from pfrl import explorers, explorer
from pfrl import utils
from pfrl.q_functions import DuelingDQN
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers
import params

tensorboard_dir_name = params.tensorboard_dir_name
tensorboard_dir = f"tensorboard/{tensorboard_dir_name}/"
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(log_dir=tensorboard_dir)

from shutil import copyfile
copyfile("params.py", tensorboard_dir+"params.py")

from DQN_model import Net, RNDNet


class RandomSelectionEpsilonGreedy(explorer.Explorer):

    def __init__(
        self,
        start_epsilon,
        end_epsilon,
        num_epsilon,
        epsilon_interval,
        random_action_func
    ):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert num_epsilon >= 0
        self.random_action_func = random_action_func
        self.epsilon = start_epsilon
        self.epsilon_interval = epsilon_interval
        self.epsilon_range = np.linspace(start_epsilon, end_epsilon, num_epsilon)

    def select_action_epsilon_greedily(self, epsilon, random_action_func, greedy_action_func):
        if np.random.rand() < epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True
    
    def compute_epsilon(self, t):
        if t%self.epsilon_interval == 0:
          return np.random.choice(self.epsilon_range)
        return self.epsilon

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, _ = self.select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        return a

    def __repr__(self):
        return "RandomSelectionEpsilonGreedy(epsilon={})".format(self.epsilon)


env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(
    params.env_name, max_frames=params.max_frames), episode_life=True, clip_rewards=True)

dueling = params.dueling
grad_norm = params.grad_norm
q_func = Net(actions=env.action_space.n, dueling=dueling)

model_copy = params.model_copy  # K
optimizer = torch.optim.Adam(q_func.parameters(), lr=params.lr, eps=0.0001)


rbuf = replay_buffers.PrioritizedReplayBuffer(
    params.per_size,
    alpha=params.per_alpha,
    beta0=params.per_beta,
    betasteps=params.per_beta_steps,
    num_steps=params.per_num_steps,
)

episodes = params.episodes
step_counter = -1
BS = params.BS
epsilon_min = params.epsilon_min
warmup = params.warmup
discount = params.discount

explorer = explorers.LinearDecayEpsilonGreedy(
    params.epsilon_max, epsilon_min, params.epsilon_steps,
    lambda: np.random.randint(env.action_space.n)
) if params.explorer_method == 0 else RandomSelectionEpsilonGreedy(
    params.epsilon_min, params.epsilon_max, params.epsilon_num, params.epsilon_interval,
    lambda: np.random.randint(env.action_space.n)
)

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255

if params.RND_reward:
    rnd_predict_model = RNDNet()
    rnd_target_model = RNDNet()
    rnd_optimizer = torch.optim.Adam(rnd_predict_model.parameters(), lr=params.lr, eps=0.0001)
    RND_models = {'target': rnd_target_model.eval().requires_grad_(False), 'predict': rnd_predict_model.train().requires_grad_(True),
                'optimizer': rnd_optimizer}  
else:
    RND_models = None

agent = DoubleDQN(
    q_func,
    optimizer,
    rbuf,
    gpu=0,
    gamma=discount,
    explorer=explorer,
    replay_start_size=warmup,
    target_update_interval=model_copy,
    minibatch_size=BS,
    clip_delta=True,
    update_interval=params.update_interval,
    batch_accumulator="mean",
    phi=phi,
    max_grad_norm=params.grad_norm,
    rnd_models=RND_models
)

train_steps = -1
episode = -1
for _ in range(params.max_steps):
  episode += 1
  obs = env.reset()
  tot_reward = 0
  tstep = 0
  done = 0
  while not done:
    tstep += 1
    if tstep % params.update_interval == 0:
      train_steps += 1
    step_counter += 1
    action = agent.act(obs)

    obs, reward, done, _ = env.step(action)
    tot_reward += reward
    reset = tstep == params.max_frames
    agent.observe(obs, reward, done, reset)
    if done or reset:
        break

  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward}')

  writer.add_scalar("Reward/Episode", tot_reward, episode)
  writer.add_scalar("Reward/Timestep", tot_reward, step_counter)
  writer.add_scalar("Reward/Trainstep", tot_reward, agent.optim_t)
  writer.add_scalar("Buffer Size/Trainstep", len(rbuf), train_steps)
  writer.add_scalar("Average Q/trainstep", agent.get_statistics()[0][1], agent.optim_t)
  writer.add_scalar("Average Loss/trainstep", agent.get_statistics()[1][1], agent.optim_t)
  writer.add_scalar("Epsilon/timestep", agent.explorer.epsilon, step_counter)
  writer.add_scalar("Epsilon/trainstep", agent.explorer.epsilon, agent.optim_t)
  if params.RND_reward:
      writer.add_scalar("Intrinsic Reward/trainstep", agent.mean_intrinsic_reward, agent.optim_t)

env.close()
writer.flush()
