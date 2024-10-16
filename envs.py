import abc
import os
import pickle
from typing import Iterable

import gymnasium
import jax
import numpy as np
import scipy.integrate
from gymnasium.spaces import Box
from matplotlib import pyplot as plt

from grn import GeneRegulatoryNetwork
from utils import create_system_rollout_module, get_memory_file


class EquationEnv(gymnasium.Env, abc.ABC):

    def __init__(self, obs_dim, action_bounds, dt=0.01):
        self.obs_dim = obs_dim
        self.dt = dt
        self.x = np.zeros(obs_dim)
        self.t = 0
        self.observation_space = Box(low=float("-inf"), high=float("inf"), shape=(self.obs_dim,))
        if isinstance(action_bounds, float):
            action_bounds = np.array([action_bounds for _ in range(obs_dim)])
        self.action_space = Box(low=-action_bounds, high=action_bounds, shape=(self.obs_dim,))

    @abc.abstractmethod
    def _step(self, t, y):
        pass

    @abc.abstractmethod
    def get_init_conditions(self):
        pass

    @abc.abstractmethod
    def get_species_names(self):
        pass

    def reset(self, *, seed=None, options=None):
        self.x = self.get_init_conditions()
        self.t = 0
        return self.x, {}

    def render(self):
        return

    def close(self):
        return


class MotionEquation(EquationEnv):

    def __init__(self, obs_dim=2, action_bounds=0.1, target=10, tol=1e-5):
        super().__init__(obs_dim, action_bounds)
        self.target = np.full(self.x.shape, fill_value=target)
        self.tol = tol

    def get_species_names(self):
        return ["x"]

    def get_init_conditions(self):
        return np.zeros(self.obs_dim)

    def _step(self, t, y):
        return

    def step(self, action):
        self.x = np.mean(scipy.integrate.odeint(func=lambda y_t, t: action,
                                                y0=self.x,
                                                t=[self.t + i / 100 for i in range(int(1 / self.dt))]), axis=0)
        self.t += 1
        reward = -np.linalg.norm(self.x - self.target)
        done = -reward < self.tol or self.t > 1000
        return self.x, reward, done, False, {}


class LotkaVolterraEquation(EquationEnv):

    def __init__(self, obs_dim=2, action_bounds=0.1, alpha=1.1, beta=0.4, gamma=0.4, delta=0.1):
        super().__init__(obs_dim, action_bounds)
        self.x = np.full(self.obs_dim, fill_value=10)
        self.a = alpha
        self.b = beta
        self.g = gamma
        self.d = delta
        self.true_x = scipy.integrate.odeint(func=lambda y_t, t: self._step(y=y_t),
                                             y0=self.x,
                                             t=[self.t + i / 100 for i in range(int(1 / self.dt) * 100)])

    def get_species_names(self):
        return ["rabbits", "foxes"]

    def get_init_conditions(self):
        return np.full(self.obs_dim, fill_value=10)

    def _step(self, t, y):
        xy = y[0] * y[1]
        return np.array([self.a * y[0] - self.b * xy, -self.g * y[1] + self.d * xy])

    def step(self, action):
        self.x = scipy.integrate.odeint(func=lambda y_t, t: self._step(y=y_t) + action,
                                        y0=self.x,
                                        t=[self.t + i / 100 for i in range(int(1 / self.dt) * 100)])
        obs = np.mean(self.x, axis=0)
        reward = - np.linalg.norm(obs - np.mean(self.true_x, axis=0))
        self.t += 100
        done = self.t >= 1000 or np.any(np.abs(obs) > 100.0) or np.any(obs < 0.0)
        self.x = self.x[-1, :]
        return obs, reward, done, False, {}

    def render(self):
        return None


class SchrodingerEquation(EquationEnv):

    def __init__(self, obs_dim=1, action_bounds=0.1, hbar=1, kx=0.1, m=1, sigma=0.1):
        super().__init__(obs_dim, action_bounds)
        self.hbar = hbar
        self.kx = kx
        self.m = m
        self.sigma = sigma
        self.a = 1.0 / (sigma * np.sqrt(np.pi))
        self.d2 = scipy.sparse.diags([1, -2, 1],
                                     [-1, 0, 1],
                                     shape=(self.x.size, self.x.size)) / 1 ** 2
        self.x_vmin = 5
        self.t = 1
        self.omega = 2 * np.pi / self.t
        self.k = self.omega ** 2 * m
        self.v = 0.5 * self.k * (self.x - self.x_vmin) ** 2
        self.t = 0
        self.x = self.get_init_conditions()

    def get_init_conditions(self):
        return np.ones(self.obs_dim)

    def _step(self, t, y):
        return -1j * (- 0.5 * self.hbar / self.m * self.d2.dot(y) + self.v / self.hbar * y)

    def get_species_names(self):
        return ["x"]

    def step(self, action):
        sol = scipy.integrate.solve_ivp(lambda y_t, t: self._step(t=t, y=y_t),
                                        t_span=[self.t, self.t + 1],
                                        y0=self.x,
                                        t_eval=np.arange(self.t, self.t + 1, self.dt),
                                        method="RK23")
        reward = 0.0
        self.t += 1
        done = self.t >= 100
        self.x = np.sqrt(np.square(sol.y.real) + np.square(sol.y.imag))[-1]
        return self.x, reward, done, False, {}


class GRNEnv(EquationEnv):

    def __init__(self, seed, obs_dim, biomodel_idx, r, ucs, cs, scale_a=10.0):
        super().__init__(obs_dim, 1.0)
        self.grn = GeneRegulatoryNetwork.create(biomodel_idx=biomodel_idx)
        self.e = pickle.load(open(os.path.join("memories", get_memory_file(biomodel_idx=biomodel_idx,
                                                                           r=r,
                                                                           ucs=ucs,
                                                                           cs=cs)), "rb"))

        self.x = self.get_init_conditions()
        self.w = self.e[1]
        self.c = self.e[2]
        self.r = int(r)
        self.ranges = [abs(self.e[3][i][1] - self.e[3][i][0]) for i in range(self.obs_dim)]
        self.ranges.pop(self.r)
        self.key = jax.random.PRNGKey(seed)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim - 1,))
        self.scale_a = scale_a

    def _step(self, t, y):
        return self.grn(key=self.key,
                        y0=y,
                        w0=self.w,
                        c=self.c)[0]

    def get_init_conditions(self):
        return self.e[0]

    def get_species_names(self):
        system = create_system_rollout_module(self.grn.config)
        return list(system.grn_step.y_indexes.keys())

    def _map_actions(self, actions: np.ndarray):
        actions = np.array([a * (r / self.scale_a) for a, r in zip(actions, self.ranges)])
        return np.insert(actions, self.r, [0.0])

    def step(self, action):
        output = self._step(t=self.t,
                            y=self.x + self._map_actions(actions=action))
        self.x = output.ys[:, -1]
        self.w = output.ws[:, -1]
        self.c = output.cs[:, -1]
        obs = np.mean(output.ys, axis=1)
        del output
        return obs, 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.w = self.e[1]
        self.c = self.e[2]
        return self.x, {}
