import gym
import numpy as np
import sys
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pfrl
from double_dqn import DoubleDQN
from pfrl import agents
from pfrl import experiments
from pfrl import explorers, explorer
from pfrl import utils
from pfrl.q_functions import DuelingDQN
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers
import params
from pfrl.replay_buffer import batch_experiences

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
    params.env_name, max_frames=params.max_frames), episode_life=False, clip_rewards=True)

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
    max_grad_norm=params.grad_norm
)

class MovingVariance():
    def __init__(self, eps=1e-10):
        self.num = 0
        self.eps = eps
    def push_val(self, x):
        self.num += 1
        if self.num == 1:
            self.old_m = x
            self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m)/self.num
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.old_m = self.new_m
            self.old_s = self.new_s
    def push(self, vec):
        try:
            if len(vec) > 0:
                for x in vec:
                    self.push_val(x)
        except TypeError:
            self.push_val(x)

    def num_samples(self):
        return self.num
    def mean(self):
        return self.new_m if self.num else 0
    def variance(self):
        return self.new_s / (self.num - 1) if self.num > 1 else 0
    def std(self):
        return (np.sqrt(self.variance()) + self.eps) if self.num > 1 else 1

class RND_module():
    def __init__(self, network_model, optimizer, lr, agent, replay_buffer):
        # this class takes the agent as argument to get parameters not for any calculations
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.rnd_predict_model = network_model().to(self.agent.device)
        self.rnd_target_model = network_model().to(self.agent.device)
        self.rnd_optimizer = optimizer(self.rnd_predict_model.parameters(), lr=lr, eps=0.0001)
        self.moving_std = MovingVariance()
        self.mean_loss = 0
    def compute_reward(self, obs):
        obs = np.array(obs)[None, ...] 
        obs = agent.phi(obs)
        obs = torch.as_tensor(obs, device=self.agent.device, dtype=torch.float32)
        rnd_targets = self.rnd_target_model.eval()(obs)
        rnd_preds = self.rnd_predict_model.eval()(obs)
        self.rnd_predict_model.train()
        rnd_rewards = ((rnd_targets - rnd_preds)**2).sum()**0.5
        rnd_rewards = rnd_rewards.item()
        self.moving_std.push_val(rnd_rewards)
        return np.clip(rnd_rewards / self.moving_std.std(), -1, 1), rnd_rewards, self.moving_std.std()
    def train(self, exp_batch):
        batch_current_states = exp_batch["state"]
        rnd_targets = self.rnd_target_model(batch_current_states)
        rnd_preds = self.rnd_predict_model(batch_current_states)
        rnd_rewards = ((rnd_targets - rnd_preds)**2).sum(axis=1)**0.5
        rnd_rewards = torch.clamp(rnd_rewards,-1,1)
        self.rnd_optimizer.zero_grad()
        mean_reward = rnd_rewards.mean()
        mean_reward.backward()
        self.rnd_optimizer.step()
        self.mean_loss = mean_reward

if params.RND_reward:
    rnd_module = RND_module(RNDNet, torch.optim.Adam, params.lr, agent, rbuf)
    agent.set_rnd_module(rnd_module)

train_steps = -1
episode = -1
for _ in range(params.max_steps):
  episode += 1
  obs = env.reset()
  tot_reward = 0
  tot_reward_ext = 0
  tot_reward_int = 0
  tstep = 0
  done = 0
  if (episode) % params.video_every == 0:
      fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
      frames = []
      int_rewards = []
  while not done:
    tstep += 1
    if (step_counter % params.update_interval == 0) and (step_counter > params.warmup):
      train_steps += 1
    #   writer.add_scalar("Total Reward/Trainstep", reward, agent.optim_t)
    #   writer.add_scalar("Average Q/trainstep", agent.get_statistics()[0][1], agent.optim_t)
    #   writer.add_scalar("Average Loss/trainstep", agent.get_statistics()[1][1], agent.optim_t)
    #   writer.add_scalar("Epsilon/trainstep", agent.explorer.epsilon, agent.optim_t)
    #   writer.add_scalar("Buffer Size/Trainstep", len(rbuf), agent.optim_t)
      if params.RND_reward:
        writer.add_scalar("RND Loss/Trainstep", rnd_module.mean_loss, agent.optim_t)
    step_counter += 1
    action = agent.act(obs)

    obs, reward, done, _ = env.step(action)
    if params.RND_reward:
        reward_int, reward_int_raw, reward_int_std = rnd_module.compute_reward(obs)
        reward_ext = reward
        reward += params.rnd_beta*reward_int
        tot_reward_ext += reward_ext
        tot_reward_int += reward_int
        # writer.add_scalar("Extrinsic Reward/Timestep", reward_ext, step_counter)
        writer.add_scalar("Intrinsic Reward/Timestep", reward_int, step_counter)
        # writer.add_scalar("Intrinsic Reward Raw/Timestep", reward_int_raw, step_counter)
        # writer.add_scalar("Intrinsic Reward STD/Timestep", reward_int_std, step_counter)
        if (episode) % params.video_every == 0:
            title = ax1.text(0.5, 1.05, f"IR {reward_int}", size=plt.rcParams["axes.titlesize"], ha="center", transform=ax1.transAxes)
            im_plt = ax1.imshow(np.array(obs)[-1], animated=True)
            int_rewards.append(reward_int)
            line_plt,  = ax2.plot(int_rewards, '-ob')
            frames.append([im_plt, title, line_plt])
    tot_reward += reward
    reset = (tstep == params.max_frames)
    agent.observe(obs, reward, done, reset)
    # writer.add_scalar("Total Reward/Timestep", reward, step_counter)
    # writer.add_scalar("Epsilon/timestep", agent.explorer.epsilon, step_counter)
    if done or reset:
        break

  if episode == 0 or episode % 10 == 0:
    print(f'finished episode {episode} at timestep {step_counter} with reward {tot_reward_ext if params.RND_reward else tot_reward}')

  writer.add_scalar("Episode Reward/Episode", tot_reward, episode)
  writer.add_scalar("Episode Reward/Trainstep", tot_reward, agent.optim_t)
  writer.add_scalar("Env Steps/Episode", step_counter, episode)
  writer.add_scalar("Train Steps/Episode", agent.optim_t, episode)
  writer.add_scalar("Buffer Size/Trainstep", len(rbuf), agent.optim_t)
  writer.add_scalar("Epsilon/trainstep", agent.explorer.epsilon, agent.optim_t)
  if params.RND_reward:
        writer.add_scalar("Episode Intrinsic Reward/Episode", tot_reward_int, episode)
        writer.add_scalar("Episode Intrinsic Reward/Trainstep", tot_reward_int, agent.optim_t)
        writer.add_scalar("Episode Extrinsic Reward/Episode", tot_reward_ext, episode)
        writer.add_scalar("Episode Extrinsic Reward/Trainstep", tot_reward_ext, agent.optim_t)
        if (episode) % params.video_every == 0:
            print('Saving Video ...........')
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
            ax1.axis('off')
            ax2.set_xlabel('step')
            ax2.set_ylabel('IR')
            ani.save(f'video/{params.env_alias}_episode_{episode}.gif')
            plt.figure(figsize=(20, 14))
            plt.plot(int_rewards, '-o')
            plt.title('Intrinsic Reward')
            plt.xlabel('step')
            plt.ylabel('IR')
            plt.savefig(f'video/{params.env_alias}_episode_{episode}.png')

env.close()
writer.flush()
