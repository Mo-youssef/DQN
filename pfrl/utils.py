import numpy as np
import pfrl
from pfrl import explorer

import gym
from gym import spaces
from gym_minigrid.wrappers import ReseedWrapper, RGBImgObsWrapper, ImgObsWrapper
from pfrl.wrappers import atari_wrappers


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
        self.epsilon_range = np.linspace(
            start_epsilon, end_epsilon, num_epsilon)

    def select_action_epsilon_greedily(self, epsilon, random_action_func, greedy_action_func):
        if np.random.rand() < epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True

    def compute_epsilon(self, t):
        if t % self.epsilon_interval == 0:
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

def mini_grid_wrapper(env_id, max_frames=0, clip_rewards=True):
    env = gym.make(env_id)
    env = ReseedWrapper(env, seeds=[0])
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    if max_frames:
        env = pfrl.wrappers.ContinuingTimeLimit(
            env, max_episode_steps=max_frames)
    # env = atari_wrappers.MaxAndSkipEnv(env, skip=0)
    env = atari_wrappers.wrap_deepmind(
        env, episode_life=False, clip_rewards=clip_rewards)
    return env


class UnswapChannel(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            env.observation_space.shape[1], env.observation_space.shape[2], env.observation_space.shape[0]), dtype=env.observation_space.dtype)
    def observation(self, frame):
        return frame.transpose(1, 2, 0)

def griddly_wrapper(env_id, max_frames=5000, clip_rewards=True):
    env = gym.make(env_id)
    env.reset()
    env = UnswapChannel(env)
    env = atari_wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=True)
    return env

def wrap_env(env_id, max_frames=5000, clip_rewards=True, episode_life=True):
    if env_id.startswith('MiniGrid'):
        env = mini_grid_wrapper(
            env_id, max_frames=max_frames, clip_rewards=True)
    elif env_id.startswith('GDY'):
        env = griddly_wrapper(
            env_id, max_frames=max_frames, clip_rewards=True)
    else:
        env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari(
            env_id, max_frames=max_frames), episode_life=True, clip_rewards=True)
    return env

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
