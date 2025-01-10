import abc
import json
import os
import pickle
import sys

import gymnasium
import jax
import numpy as np
import scipy.integrate
from gymnasium.spaces import Box

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

    def __init__(self, seed, exp, obs_dim, grn, r, idx, scale_a=1.0):
        super().__init__(obs_dim, 1.0)
        self.grn = grn
        self.key = jax.random.PRNGKey(seed)
        self.mem_data = pickle.load(
            open(os.path.join("memories", exp, get_memory_file(biomodel_idx=grn.biomodel_idx,
                                                               r=r,
                                                               idx=idx)), "rb"))
        self.r = int(r)
        self.s = self.mem_data[10 if self.mem_data[10] is not None else 9]
        self.ranges = [abs(self.mem_data[3][i][1] - self.mem_data[3][i][0]) for i in range(self.obs_dim)]
        self.ranges.pop(self.r)
        self.mean_relax = self.mem_data[4]
        self.response_reg = self.mem_data[5]
        self.stimulus_reg = self.mem_data[6]  # TODO: dataclass for all this stuff
        # self.prev_e = self.mem_data[0]  # self._recoup()
        self.prev_mean = None
        # self.mem_data[0] = self.prev_e.ys[:, -1]
        self.x = self.get_init_conditions()
        self.w = self.mem_data[1]
        self.c = self.mem_data[2]
        self.r_scale = self.mem_data[7]
        self.action_space = Box(low=-1.0, high=1.0, shape=((self.obs_dim - 2) * 2,), seed=seed)
        self.scale_a = scale_a
        self.obs = None
        self.action_map = self._build_action_map()
        self.grn.set_time(500)  # 4000)
        self.grn.NUM_PULSES = 1
        self.exp = exp
        self.rewards = [self.mean_relax]

    def _build_action_map(self):
        m = {}
        i = 0
        for sp in range(len(self.get_species_names())):
            if sp != self.r and sp != self.s:
                m[i] = sp
                i += 1
        return m

    def _recoup(self, n_stim=4, increment=500):
        r = self.grn(key=self.key)[0]
        y0 = r.ys[:, -1]
        w0 = r.ws[:, -1]
        n_secs = 2500
        for i in range(0, n_stim * 2, 2):
            e = self.grn.stimulate(key=self.key,
                                   y0=y0,
                                   w0=w0,
                                   t0=n_secs,
                                   stimulus={self.s: self.mem_data[3][self.s][self.stimulus_reg % 2]})
            n_secs += e.ys.shape[1] * self.grn.config.deltaT
            r = self.grn(key=self.key,
                         t0=n_secs,
                         y0=e.ys[:, -1],
                         w0=e.ws[:, -1])[0]
            n_secs += r.ys.shape[1] * self.grn.config.deltaT
            y0 = r.ys[:, -1]
            w0 = r.ws[:, -1]
            self.grn.set_time(n_secs=2500 + ((i // 2 + 1) * increment))
        return r

    def _step(self, t, y, actions):
        stimuli = {self.s: self.mem_data[3][self.s][self.stimulus_reg % 2]}
        for control, value in actions.items():
            stimuli[control] = value
        return self.grn.stimulate(key=self.key,
                                  y0=y,
                                  w0=self.w,
                                  t0=t,
                                  stimulus=stimuli)

    def get_init_conditions(self):
        return self.mem_data[0]

    def get_species_names(self):
        system = create_system_rollout_module(self.grn.config)
        return list(system.grn_step.y_indexes.keys())

    def get_action_names(self):
        species = self.get_species_names()
        old_s = species[self.s]
        species.remove(species[self.r])
        species.remove(old_s)
        return species

    def _map_actions(self, actions: np.ndarray):
        actions = np.array([a * (r / self.scale_a) for a, r in zip(actions, self.ranges)])
        return np.insert(actions, self.r, [0.0])

    def _get_reward(self, output):
        mean = np.nanmean(output.ys[self.r, :])
        if (self.exp == "habit" and self.response_reg == 1) or (self.exp == "sens" and self.response_reg == 2):
            # prev_mean = np.mean(self.prev_e.ys[self.r, :])
            # return mean - (prev_mean / 1.5)
            return (mean - self.prev_mean[self.r]) / self.prev_mean[self.r]
        elif (self.exp == "sens" and self.response_reg == 1) or (self.exp == "habit" and self.response_reg == 2):
            # prev_mean = np.mean(self.prev_e.ys[self.r, :])
            # return (prev_mean * 1.5) - mean
            return (self.prev_mean[self.r] - mean) / self.prev_mean[self.r]
        # return -mean if self.response_reg == 1 else mean
        return (mean - self.prev_mean[self.r]) / self.prev_mean[self.r] \
            if self.response_reg == 1 else (self.prev_mean[self.r] - mean) / self.prev_mean[self.r]

    def step(self, action):
        if isinstance(action, np.ndarray):
            idx = np.argmax(action) // 2
            action = {self.action_map[idx]: self.mem_data[3][idx][np.argmax(action) % 2]}
        output = self._step(t=self.t,
                            y=self.x,
                            actions=action)
        self.x = output.ys[:, -1]
        self.w = output.ws[:, -1]
        self.c = output.cs[:, -1]
        self.obs = output.ys  # np.mean(output.ys, axis=1)
        r = self._get_reward(output=output)
        reward = (r - np.median(self.rewards)) / (np.std(self.rewards) if len(self.rewards) > 1 else 1.0)
        print(r, reward)
        self.rewards.append(r)
        self.t += output.ys.shape[1] * self.dt
        info = self._get_info()
        del output
        self.prev_mean = np.nanmean(self.obs, axis=1)
        # print(self.prev_mean)
        return (self.prev_mean,
                reward,
                self.t >= 12500,  # 25500 + (4000 * 5),
                np.any(np.isnan(self.obs)),
                info)

    def _is_not_memory(self, e, response, response_reg):
        if e is None:
            return False
        elif (self.exp == "habit" and self.response_reg == 1) or (self.exp == "sens" and self.response_reg == 2):
            prev_mean = np.mean(self.prev_e.ys[self.r, :])
            return np.mean(e[response, :]) > (prev_mean / 1.5)
        elif (self.exp == "sens" and self.response_reg == 1) or (self.exp == "habit" and self.response_reg == 2):
            prev_mean = np.mean(self.prev_e.ys[self.r, :])
            return np.mean(e[response, :]) < (prev_mean * 1.5)
        elif response_reg == 1:
            return np.mean(e[response, :]) < self.r_scale * self.mean_relax
        return np.mean(e[response, :]) > self.mean_relax / self.r_scale

    def _get_info(self):
        return {"full_obs": self.obs, "is_broken": self._is_not_memory(e=self.obs,
                                                                       response=self.r,
                                                                       response_reg=self.response_reg)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.t = 10000  # 25500
        self.w = self.mem_data[1]
        self.c = self.mem_data[2]
        self.prev_mean = self.x
        return self.x, self._get_info()


if __name__ == "__main__":
    seed = sys.argv[1]
    env = pickle.load(open(os.path.join("envs", ".".join([str(seed), "pickle"])), "rb"))
    with open(os.path.join("pop", sys.argv[4], ".".join([sys.argv[2], "json"])), "w") as f:
        d = env.step(action={int(i.split(",")[0]): float(i.split(",")[1]) for i in sys.argv[3].split("/")})
        json.dump({"r": d[1].item(),
                   "terminated": bool(d[2]),
                   "truncated": bool(d[3]),
                   "is_broken": bool(d[-1]["is_broken"])}, f)
