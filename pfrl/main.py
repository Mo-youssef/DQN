import gym
import numpy as np
import sys
import pickle
import os
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import pfrl
from double_dqn import DoubleDQN
from categorical_double_dqn import CategoricalDoubleDQN
from pfrl import agents
from pfrl import nn as pnn
from pfrl import experiments
from pfrl import explorers, explorer
from pfrl import utils
from pfrl.q_functions import DuelingDQN
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers
import params
from pfrl.replay_buffer import batch_experiences
from pfrl.q_functions import DistributionalDuelingDQN

import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

tensorboard_dir_name = params.tensorboard_dir_name
tensorboard_dir = f'tensorboard/{tensorboard_dir_name}/'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(log_dir=tensorboard_dir)

from shutil import copyfile
copyfile("params.py", tensorboard_dir+"params.py")

from DQN_model import Net, RNDNet, Embedding_fn, Embedding_full
from IR_modules import NGU_module, RND_module
from utils import RandomSelectionEpsilonGreedy, wrap_env

import pdb

env = wrap_env(params.env_name, max_frames=params.max_frames, clip_rewards=True)

if params.rainbow:
    q_func = DistributionalDuelingDQN(env.action_space.n, params.n_atoms, params.v_min, params.v_max) 
else:
    q_func = Net(actions=env.action_space.n, dueling=params.dueling)
    
if params.noisynet:
    pnn.to_factorized_noisy(q_func, sigma_scale=params.noisy_net_sigma)
    explorer = explorers.Greedy()
else:
    explorer = explorers.LinearDecayEpsilonGreedy(
        params.epsilon_max, params.epsilon_min, params.epsilon_steps,
        lambda: np.random.randint(env.action_space.n)
    ) if params.explorer_method == 0 else RandomSelectionEpsilonGreedy(
        params.epsilon_min, params.epsilon_max, params.epsilon_num, params.epsilon_interval,
        lambda: np.random.randint(env.action_space.n)
    )

optimizer = torch.optim.Adam(q_func.parameters(), lr=params.lr, eps=1e-08) # eps=1.5*10**-4)


rbuf = replay_buffers.PrioritizedReplayBuffer(
    params.per_size,
    alpha=params.per_alpha,
    beta0=params.per_beta,
    betasteps=params.per_beta_steps,
    num_steps=params.per_num_steps,
    normalize_by_max="memory"
)

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255
if params.rainbow:
    agent = CategoricalDoubleDQN(
        q_func,
        optimizer,
        rbuf,
        gpu=0,
        gamma=params.discount,
        explorer=explorer,
        minibatch_size=params.BS,
        replay_start_size=params.warmup,
        target_update_interval=params.model_copy,
        clip_delta=True,
        update_interval=params.update_interval,
        batch_accumulator="mean",
        phi=phi,
        max_grad_norm=params.grad_norm
    )
else:
    agent = DoubleDQN(
        q_func,
        optimizer,
        rbuf,
        gpu=0,
        gamma=params.discount,
        explorer=explorer,
        replay_start_size=params.warmup,
        target_update_interval=params.model_copy,
        minibatch_size=params.BS,
        clip_delta=True,
        update_interval=params.update_interval,
        batch_accumulator="mean",
        phi=phi,
        max_grad_norm=params.grad_norm
    )


if params.RND_reward:
    rnd_module = RND_module(RNDNet, torch.optim.Adam, params.lr, agent)
    agent.set_rnd_module(rnd_module)
if params.NGU_reward:
    ngu_module = NGU_module(Embedding_fn, Embedding_full, torch.optim.Adam, params.lr, agent,
                            env.action_space.n, params.max_frames, params.ngu_embed_size, params.ngu_k_neighbors)

step_counter = -1
ir_train_steps = -1
episode = -1
while step_counter < params.max_steps:
    episode += 1
    obs = env.reset()
    tot_reward = 0
    tot_reward_ext = 0
    tot_reward_int = 0
    tstep = 0
    done = 0
    if params.NGU_reward:
        ngu_module.reset()
    if (episode) % params.video_every == 0:
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        frames = []
        int_rewards = []
    while not done:

        tstep += 1
        step_counter += 1
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)

        # Training IR modules
        # if (step_counter % params.update_interval == 0) and (step_counter > params.warmup):
        if step_counter > params.ir_warmup:
            ir_train_steps += 1
            if params.NGU_reward and (step_counter % params.ngu_update_interval == 0):
                ngu_module.train()
            if params.RND_reward and (step_counter % params.update_interval == 0):
                # rnd_module.train()
                writer.add_scalar("RND Loss/Trainstep", rnd_module.mean_loss, agent.optim_t)

        # adding intrinsic reward to extrinsic reward
        if params.IR_reward:
            # pdb.set_trace()
            if params.RND_reward:
                rnd_reward = rnd_module.compute_reward(obs)
                reward_int = rnd_reward
            if params.NGU_reward:
                ngu_reward = ngu_module.compute_reward(obs)
                reward_int = ngu_reward
            if params.RND_reward and params.NGU_reward:
                reward_int = ngu_reward * min(max(1+rnd_reward, 1), params.ngu_L)
            reward_ext = reward
            reward += params.ir_beta*reward_int if (step_counter > params.warmup) else 0
            tot_reward_ext += reward_ext
            tot_reward_int += reward_int
            
        tot_reward += reward
        reset = (tstep == params.max_frames)
        agent.observe(obs, reward, done, reset)

        if done or reset:
            break

########################################################################## Logging Data and Plotting calls ##########################################################################

        if params.IR_reward:
            writer.add_scalar("Intrinsic Reward/Timestep", reward_int, step_counter)
            if (episode) % params.video_every == 0:
                title = ax1.text(0.5, 1.05, f"IR {reward_int}", size=plt.rcParams["axes.titlesize"], ha="center", transform=ax1.transAxes)
                im_plt = ax1.imshow(np.array(obs)[-1], animated=True)
                int_rewards.append(reward_int)
                line_plt,  = ax2.plot(int_rewards, '-ob')
                frames.append([im_plt, title, line_plt])
        elif (episode) % params.video_every == 0:
            im_plt = ax1.imshow(np.array(obs)[-1], animated=True)
            frames.append([im_plt])

        if step_counter % params.per_size == 0:
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss)
            with open('mem_usage.txt', 'a') as f:
                f.write(str(process.memory_info().rss) + '\n')

    if episode == 0 or episode % 10 == 0:
        print(
            f'finished episode {episode} at timestep {step_counter} with reward {tot_reward_ext if params.IR_reward else tot_reward}')

    writer.add_scalar("Episode Reward/Episode", tot_reward, episode)
    writer.add_scalar("Episode Reward/Trainstep", tot_reward, agent.optim_t)
    writer.add_scalar("Env Steps/Episode", step_counter, episode)
    writer.add_scalar("Train Steps/Episode", agent.optim_t, episode)
    writer.add_scalar("Buffer Size/Trainstep", len(rbuf), agent.optim_t)
    if params.IR_reward:
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
            print('Done ...........')
    elif (episode) % params.video_every == 0:
            print('Saving Video ...........')
            ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
            ax1.axis('off')
            ani.save(f'video/{params.env_alias}_episode_{episode}.gif')
            print('Done ...........')


env.close()
writer.flush()

#   writer.add_scalar("Average Q/trainstep", agent.get_statistics()[0][1], agent.optim_t)
#   writer.add_scalar("Average Loss/trainstep", agent.get_statistics()[1][1], agent.optim_t)
    #   writer.add_scalar("Epsilon/trainstep", agent.explorer.epsilon, agent.optim_t)
        # writer.add_scalar("Total Reward/Timestep", reward, step_counter)
        # writer.add_scalar("Epsilon/timestep", agent.explorer.epsilon, step_counter)
# writer.add_scalar("Intrinsic Reward Raw/Timestep", reward_int_raw, step_counter)
# writer.add_scalar("Intrinsic Reward STD/Timestep", reward_int_std, step_counter)
# writer.add_scalar("Extrinsic Reward/Timestep", reward_ext, step_counter)
