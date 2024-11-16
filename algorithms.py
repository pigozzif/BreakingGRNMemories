import random

import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

from al import AssociativeLearning


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
            control = env.action_map[action // 2]
            obs, reward, terminated, truncated, info = env.step({control:
                                                                 env.mem_data[3][control][action % 2]})
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


class GeneticAlgorithmCombinatorics(BaseAlgorithm):

    def __init__(self, env, seed, pop_size=20, num_evals=1000):
        super().__init__(None, env, 0.0, seed=seed)
        self.solutions = np.zeros((pop_size, self.env.envs[0].action_space.shape[0]))
        self.fitness_list = np.zeros(pop_size)
        self.cache = set()
        self.env.envs[0].reset()
        self.num_sols = 3 ** self.env.envs[0].action_space.shape[0]
        self._create_pop(pop_size=pop_size)
        self.num_evals = num_evals
        self.best = None

    def _create_pop(self, pop_size):
        i = 0
        while i < min(pop_size, self.num_sols) and len(self.cache) < self.num_sols:
            individual = np.random.randint(0, 3, self.env.envs[0].action_space.shape[0], dtype=np.int32)
            if tuple(individual) not in self.cache:
                self.solutions[i] = individual
                self.fitness_list[i] = self._evaluate_action(action=individual)
                self.cache.add(tuple(individual))
                i += 1

    def _map_action(self, action):
        return {self.env.envs[0].action_map[i]: self.env.envs[0].mem_data[3][self.env.envs[0].action_map[i]][a]
                for i, a in enumerate(action) if a != 2}

    def _evaluate_action(self, action):
        obs, reward, terminated, truncated, info = self.env.envs[0].step(action=self._map_action(action=action))
        if terminated or truncated:
            self.env.envs[0].reset()
        if truncated or np.isnan(reward):
            reward = float("-inf")
        return reward

    def _tournament_select(self, k, n=5):
        selected = []
        for _ in range(k):
            contenders = np.random.choice(np.arange(len(self.solutions)), size=n, replace=False)
            selected.append(self.solutions[np.argmax(self.fitness_list[contenders])])
        return selected

    def _cx(self):
        parents = self._tournament_select(k=2)
        child = parents[0].copy()
        for k, (i, j) in enumerate(zip(parents[0], parents[1])):
            child[k] = i if random.random() > 0.5 else j
        return self._mutation(parent=child)

    def _mutation(self, parent=None):
        child = self._tournament_select(k=1)[0].copy() if parent is None else parent
        child[random.randint(0, len(child) - 1)] = random.randint(0, 2)
        return child

    def _reproduce(self):
        children, fitness = [], []
        for _ in range(len(self.solutions)):
            if random.random() > 0.8:
                child = self._mutation()
            else:
                child = self._cx()
            if tuple(child) not in self.cache:
                children.append(child)
                fitness.append(self._evaluate_action(action=child))
                self.cache.add(tuple(child))
        # print(min([child[:int(len(child) // 2)] for child in children]),
        #       max([child[:int(len(child) // 2)] for child in children]))
        return children, fitness

    def _survival_select(self, offspring, fitness_list):
        if not offspring or not fitness_list:
            return
        self.fitness_list = np.concatenate([self.fitness_list, fitness_list], axis=0)
        survival_idx = np.argsort(self.fitness_list)[-len(self.solutions):]
        self.solutions = np.concatenate([self.solutions, offspring], axis=0)[survival_idx]
        self.fitness_list = self.fitness_list[survival_idx]

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True,
              progress_bar=False):
        i = 0
        while i < total_timesteps and (len(self.cache) < self.num_evals and len(self.cache) < self.num_sols):
            offspring, fitness = self._reproduce()
            self._survival_select(offspring=offspring, fitness_list=fitness)
            i += 1
        self.best = self.solutions[np.argmax(self.fitness_list)]
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


class GeneticAlgorithmNumerical(GeneticAlgorithmCombinatorics):

    def __init__(self, env, seed, pop_size=20, num_evals=1000, sigma=0.1):
        super().__init__(env, seed, pop_size, num_evals)
        self.solutions = np.zeros((pop_size, self.env.envs[0].action_space.shape[0] * 2))
        self.sigma = sigma
        self.num_sols = float("inf")
        self.bounds = self.env.envs[0].mem_data[0]
        for i in range(pop_size):
            individual = np.hstack([np.random.uniform(low=0.0,
                                                      high=6.0,  # self.sigma,
                                                      size=self.env.envs[0].action_space.shape[0]),
                                    np.random.randint(low=0,
                                                      high=2,
                                                      size=self.env.envs[0].action_space.shape[0])])
            if tuple(individual) not in self.cache:
                self.solutions[i] = individual
                self.fitness_list[i] = self._evaluate_action(action=individual)
                self.cache.add(tuple(individual))

    def _create_pop(self, pop_size):
        return

    def _map_action(self, action):
        # print(len(self.cache))
        mask = action[-len(action) // 2:]
        return {self.env.envs[0].action_map[i]: self.bounds[self.env.envs[0].action_map[i]] * np.exp(a)
                for i, a in enumerate(action[: len(action) // 2]) if mask[i]}

    def _mutation(self, parent=None):
        child = self._tournament_select(k=1)[0].copy() if parent is None else parent
        idx = random.randint(0, len(child) - 1)
        if idx < len(child) // 2:
            child[idx] += np.random.normal(loc=0.0, scale=self.sigma, size=1)
        else:
            child[idx] = 1 - child[idx]
        return child
