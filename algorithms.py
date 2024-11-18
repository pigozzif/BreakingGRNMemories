import json
import os
import pickle
import random
import time
from multiprocessing import Pool

import numpy as np
from gymnasium.spaces import Discrete
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
            control = env.action_map[action // 2]
            obs, reward, terminated, truncated, info = env.step({control:
                                                                     env.mem_data[3][control][action % 2]})
            if reward > self.best_reward:
                self.best = action
                self.best_reward = reward
            if terminated or truncated:
                env.reset()
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return np.array(self.best), None

    def _setup_model(self):
        return

    def save(self, path, exclude=None, include=None):
        return

    def load(self, path, env=None, device="auto", custom_objects=None, print_system_info=False, force_reset=True,
             **kwargs):
        return self


def calc_fitness(args):
    idx, action, seed, dir_name = args
    if not action:
        return idx, float("-inf"), False
    os.system("python envs.py {0} {1} {2} {3}".format(seed,
                                                      idx,
                                                      "/".join([",".join([str(k), str(v)])
                                                                for k, v in action.items()]),
                                                      dir_name))
    data = json.load(open(os.path.join("pop", dir_name, ".".join([str(idx), "json"]))))
    if data["truncated"] or np.isnan(data["r"]):
        return idx, float("-inf"), False
    return idx, data["r"], data["is_broken"]


class GeneticAlgorithmCombinatorics(BaseAlgorithm):

    def __init__(self, env, seed, file_name, pop_size=20, num_evals=1000, num_workers=8):
        super().__init__(None, env, 0.0, seed=seed)
        self.idx = 0
        self.cache = set()
        self.env.envs[0].reset()
        self.num_sols = 3 ** self.env.envs[0].action_space.shape[0]
        self.solutions = np.zeros((min(pop_size, self.num_sols), self.env.envs[0].action_space.shape[0]))
        pickle.dump(self.env.envs[0].env, open(os.path.join("envs", ".".join([str(seed), "pickle"])), "wb"))
        self._create_pop(pop_size=pop_size)
        self.start = time.time()
        self._create_file(file_name=file_name)
        self.dir_name = self.file_name.replace(".csv", "").replace("output/", "")
        self.num_workers = num_workers
        self._create_pop(pop_size=pop_size)
        self.num_evals = num_evals
        self.best = None
        self.idx = min(pop_size, self.num_sols)

    def _create_pop(self, pop_size):
        i = 0
        while i < min(pop_size, self.num_sols) and len(self.cache) < self.num_sols:
            individual = np.random.randint(0, 3, self.env.envs[0].action_space.shape[0], dtype=np.int32)
            if tuple(individual) not in self.cache:
                self.solutions[i] = individual
                self.cache.add(tuple(individual))
                i += 1
        self.fitness_list = self._parallel_eval(individuals=self.solutions)

    def _create_file(self, file_name):
        self.file_name = os.path.join("output", file_name)
        with open(self.file_name, "w") as file:
            file.write("#{\"t_start\": 1730254023.772545, \"env_id\": \"None\"}\nr,l,t,is_broken\n")

    def _record(self, results):
        with open(self.file_name, "a") as file:
            for _, r, is_broken in results:
                file.write(",".join([str(r), "1", str(time.time() - self.start), str(is_broken)]) + "\n")

    def _map_action(self, action):
        return {
            self.env.envs[0].env.action_map[i]: self.env.envs[0].env.mem_data[3][self.env.envs[0].env.action_map[i]][
                int(a)]
            for i, a in enumerate(action) if int(a) != 2}

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
        children = []
        for _ in range(len(self.solutions)):
            if random.random() > 0.8:
                child = self._mutation()
            else:
                child = self._cx()
            if tuple(child) not in self.cache:
                children.append(child)
                self.cache.add(tuple(child))
        return children

    def _survival_select(self, offspring, fitness_list):
        if not len(offspring) or not len(fitness_list):
            return
        self.fitness_list = np.concatenate([self.fitness_list, fitness_list], axis=0)
        survival_idx = np.argsort(self.fitness_list)[-len(self.solutions):]
        self.solutions = np.concatenate([self.solutions, offspring], axis=0)[survival_idx]
        self.fitness_list = self.fitness_list[survival_idx]

    def _parallel_eval(self, individuals):
        with Pool(self.num_workers) as pool:
            results = pool.map(calc_fitness, [(self.idx + j, self._map_action(action=child), self.seed, self.dir_name)
                                              for j, child in enumerate(individuals)])
        self._record(results=results)
        return np.array([r[1] for r in sorted(results, key=lambda x: x[0])])

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True,
              progress_bar=False):
        i = 0
        while i < total_timesteps and (len(self.cache) < self.num_evals and len(self.cache) < self.num_sols):
            offspring = self._reproduce()
            fitness = self._parallel_eval(individuals=offspring)
            self.idx += len(offspring)
            self._survival_select(offspring=offspring, fitness_list=fitness)
            i += 1
        self.best = self.solutions[np.argmax(self.fitness_list)]
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return np.array(self.best), None

    def _setup_model(self):
        return

    def save(self, path, exclude=None, include=None):
        return

    def load(self, path, env=None, device="auto", custom_objects=None, print_system_info=False, force_reset=True,
             **kwargs):
        return self


class GeneticAlgorithmNumerical(GeneticAlgorithmCombinatorics):

    def __init__(self, env, seed, file_name, pop_size=20, num_evals=20, num_workers=1, sigma=0.1):
        super().__init__(env, seed, file_name=file_name, pop_size=pop_size, num_evals=num_evals, num_workers=num_workers)
        self.solutions = np.zeros((pop_size, self.env.envs[0].action_space.shape[0] * 2))
        self.sigma = sigma
        self.num_sols = float("inf")
        self.bounds = self.env.envs[0].mem_data[0]
        self.idx = 0
        for i in range(pop_size):
            individual = np.hstack([np.random.uniform(low=0.0,
                                                      high=6.0,  # self.sigma,
                                                      size=self.env.envs[0].action_space.shape[0]),
                                    np.random.randint(low=0,
                                                      high=2,
                                                      size=self.env.envs[0].action_space.shape[0])])
            if tuple(individual) not in self.cache:
                self.solutions[i] = individual
                self.cache.add(tuple(individual))
        self.fitness_list = self._parallel_eval(individuals=self.solutions)
        self.idx += pop_size

    def _create_pop(self, pop_size):
        return

    def _map_action(self, action):
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
