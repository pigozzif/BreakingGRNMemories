from itertools import product

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm


class SingleExhaustiveSolver(BaseAlgorithm):

    def __init__(self, env, seed):
        super().__init__(None, env, 0.0, seed=seed)
        self.best = None
        self.best_reward = float("-inf")
        env.env.action_space = Discrete(n=(env.env.obs_dim - 2) * 2, seed=seed)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True,
              progress_bar=False):
        env = self.env.envs[0]
        env.reset()
        for action in tqdm(range(env.env.action_space.n)):
            obs, reward, terminated, truncated, info = env.step(action)
            if reward > self.best_reward:
                self.best = action
                self.best_reward = reward
            if terminated or truncated:
                env.reset()
        return self

    def predict(self, observation, state=None,  episode_start=None, deterministic=False):
        return np.array(self.best), None

    def _setup_model(self):
        return

    def save(self, path, exclude=None, include=None):
        return

    def load(self, path, env=None, device="auto", custom_objects=None, print_system_info=False, force_reset=True,
             **kwargs):
        return self


class MultipleExhaustiveSolver(BaseAlgorithm):

    def __init__(self, env, seed):
        super().__init__(None, env, 0.0, seed=seed)
        self.best = None
        self.best_reward = float("-inf")
        env.env.action_space = MultiDiscrete(nvec=[3 for _ in range(env.env.obs_dim - 2)], seed=seed)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True,
              progress_bar=False):
        env = self.env.envs[0]
        env.reset()
        possible_actions = product([0, 1, 2], repeat=len(env.get_action_names()))
        for actions in tqdm(possible_actions):
            obs, reward, terminated, truncated, info = env.step([(i * 2) + a for i, a in enumerate(actions) if a != 2])
            if reward > self.best_reward:
                self.best = actions
                self.best_reward = reward
            if terminated or truncated:
                env.reset()
        return self

    def predict(self, observation, state=None,  episode_start=None, deterministic=False):
        return np.array(self.best), None

    def _setup_model(self):
        return

    def save(self, path, exclude=None, include=None):
        return

    def load(self, path, env=None, device="auto", custom_objects=None, print_system_info=False, force_reset=True,
             **kwargs):
        return self
