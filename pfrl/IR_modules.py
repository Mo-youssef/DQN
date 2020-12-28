import numpy as np
from collections import deque

import torch
import torch.nn as nn

import params
from pfrl.replay_buffer import batch_experiences

from DQN_model import Net, RNDNet
from utils import MovingVariance

# import pdb


class RND_module():
    def __init__(self, network_model, optimizer, lr, agent):
        # this class takes the agent as argument to get parameters not for any calculations
        self.agent = agent
        self.rnd_predict_model = network_model().to(self.agent.device)
        self.rnd_target_model = network_model().to(self.agent.device)
        self.rnd_optimizer = optimizer(
            self.rnd_predict_model.parameters(), lr=lr, eps=1e-08)
        self.moving_std = MovingVariance()
        self.mean_loss = 0

    def compute_reward(self, obs):
        obs = np.array(obs)[None, ...]
        obs = self.agent.phi(obs)
        obs = torch.as_tensor(obs, device=self.agent.device,
                              dtype=torch.float32)[:, -1:, :, :]
        rnd_targets = self.rnd_target_model.eval()(obs)
        rnd_preds = self.rnd_predict_model.eval()(obs)
        self.rnd_predict_model.train()
        rnd_rewards = ((rnd_targets - rnd_preds)**2).sum()
        rnd_rewards = rnd_rewards.item()
        self.moving_std.push_val(rnd_rewards)
        if params.rnd_clip:
            ret_reward = np.clip(rnd_rewards / self.moving_std.std(), -1, 1)
        else:
            ret_reward = rnd_rewards / self.moving_std.std()
        return ret_reward  # , rnd_rewards, self.moving_std.std()

    def train(self, exp_batch=None):
        if exp_batch is None:
            transitions = self.agent.replay_buffer.uniform_sample(
                self.agent.minibatch_size)
            exp_batch = batch_experiences(
                transitions,
                device=self.agent.device,
                phi=self.agent.phi,
                gamma=self.agent.gamma,
                batch_states=self.agent.batch_states,
            )
        batch_current_states = exp_batch["state"][:, -1:, :, :]
        rnd_targets = self.rnd_target_model(batch_current_states)
        rnd_preds = self.rnd_predict_model(batch_current_states)
        rnd_rewards = ((rnd_targets - rnd_preds)**2).sum(axis=1)
        if params.rnd_clip:
            rnd_rewards = torch.clamp(rnd_rewards, -1, 1)
        self.rnd_optimizer.zero_grad()
        mean_reward = rnd_rewards.mean()
        mean_reward.backward()
        self.rnd_optimizer.step()
        self.mean_loss = mean_reward


class EpisodicMemory():
    def __init__(self, max_size, embedding_size, k_neighbors, eps=1e-3, C=0.01, psi=1e-5):
        self.num = 0
        self.max_size = max_size
        self.memory = np.zeros((max_size, embedding_size))
        self.k_neighbors = k_neighbors
        self.running_sum = 0
        self.running_num = 0
        self.eps = eps
        self.C = C
        self.psi = psi
        self.max_sim = np.sqrt(self.k_neighbors) - 10 * self.C

    def reset(self):
        self.num = 0
        self.running_sum = 0
        self.running_num = 0

    def add_item(self, embedding):
        self.memory[self.num % self.max_size] = np.array(embedding).ravel()
        self.num = (self.num + 1) % self.max_size

    def score(self, embedding):
        if self.num < self.k_neighbors:
            return 1 
        test = np.array(embedding).ravel()[None, :]
        dists = np.sum((test - self.memory[:self.num])**2, axis=1)
        k_dist = np.partition(dists, self.k_neighbors+1)[:self.k_neighbors+1] if len(
            dists) > self.k_neighbors+1 else np.sort(dists)[:self.k_neighbors+1]
        k_dist = np.delete(k_dist, k_dist.argmin()) if len(
            k_dist) > 1 else k_dist
        self.running_sum += np.sum(k_dist)
        self.running_num += len(k_dist)
        running_mean = self.running_sum / self.running_num
        dist_normalized = k_dist / max(running_mean, self.psi)   #(running_mean if abs(running_mean - 0)>self.psi else self.psi )
        dist_normalized = np.maximum(dist_normalized - self.psi, 0)
        dist_kernel = self.eps / (self.eps + dist_normalized)
        sim = np.sqrt(np.sum(dist_kernel)) + self.C
        return 0 if sim >= self.max_sim else 1/sim


class NGU_module():
    def __init__(self, embedding_fn, embedding_model, optimizer,
                 lr, agent, n_actions, episodic_max_size, embedding_size, k_neighbors):
        # this class takes the agent as argument to get parameters not for any calculations
        self.agent = agent
        self.embedding_fn = embedding_fn(
            embedding_size=embedding_size).to(self.agent.device)
        self.embedding_model = embedding_model(
            self.embedding_fn, n_actions).to(self.agent.device)
        self.optimizer = optimizer(
            self.embedding_model.parameters(), lr=lr, eps=1e-08)
        self.episodic_memory = EpisodicMemory(
            episodic_max_size, embedding_size, k_neighbors)
        self.loss_fn = nn.CrossEntropyLoss()

    def reset(self):
        self.episodic_memory.reset()

    def compute_reward(self, obs):
        # get embedding
        obs = np.array(obs)[None, ...]
        obs = self.agent.phi(obs)
        obs = torch.as_tensor(obs, device=self.agent.device,
                              dtype=torch.float32)[:, -1:, :, :]
        with torch.no_grad():
            embedding = self.embedding_fn.eval()(obs)
        self.embedding_fn.train()
        embedding = embedding.cpu().detach().numpy()

        # add memory to episodic memory and get score
        self.episodic_memory.add_item(embedding)
        reward = self.episodic_memory.score(embedding)

        # clipping reward
        # if params.ngu_clip:
        #     reward = np.clip(reward, -1, 1)
        # else:
        #     reward = reward
        return reward

    def train(self, exp_batch=None):
        if exp_batch is None:
            transitions = self.agent.replay_buffer.uniform_sample(
                self.agent.minibatch_size)
            exp_batch = batch_experiences(
                transitions,
                device=self.agent.device,
                phi=self.agent.phi,
                gamma=self.agent.gamma,
                batch_states=self.agent.batch_states,
            )
        batch_current_states = exp_batch["state"][:, -1:, :, :]
        batch_next_states = exp_batch["true_next_state"][:, -1:, :, :]
        batch_actions = exp_batch["action"]
        output = self.embedding_model(batch_current_states, batch_next_states)
        loss = self.loss_fn(output, batch_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
