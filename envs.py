import json
import os
import pickle
import sys

import gymnasium
import jax
import numpy as np
from gymnasium.spaces import Box

from utils import create_system_rollout_module, get_memory_file


class GRNEnv(gymnasium.Env):

    def __init__(self, seed, exp, obs_dim, grn, r, idx, scale_a=1.0, dt=0.01):
        super().__init__()
        self.obs_dim = obs_dim
        self.dt = dt
        self.t = 0
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
        self.prev_mean = self.mem_data[-1]
        self.x = self.get_init_conditions()
        self.w = self.mem_data[1]
        self.c = self.mem_data[2]
        self.r_scale = self.mem_data[7]
        self.observation_space = Box(low=float("-inf"), high=float("inf"), shape=(self.obs_dim,))
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

    def _get_reward(self, output, eps=1e-10):
        mean = np.nanmean(output.ys[self.r, :])
        if (self.exp == "habit" and self.response_reg == 1) or (self.exp == "sens" and self.response_reg == 2):
            # prev_mean = np.mean(self.prev_e.ys[self.r, :])
            # return mean - (prev_mean / 1.5)
            return (mean - self.prev_mean[self.r]) / (self.prev_mean[self.r] + eps)
        elif (self.exp == "sens" and self.response_reg == 1) or (self.exp == "habit" and self.response_reg == 2):
            # prev_mean = np.mean(self.prev_e.ys[self.r, :])
            # return (prev_mean * 1.5) - mean
            return (self.prev_mean[self.r] - mean) / (self.prev_mean[self.r] + eps)
        # return -mean if self.response_reg == 1 else mean
        return (mean - self.prev_mean[self.r]) / (self.prev_mean[self.r] + eps) \
            if self.response_reg == 1 else (self.prev_mean[self.r] - mean) / (self.prev_mean[self.r] + eps)

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
        if np.isnan(r):
            r = 0.0
        reward = (r - np.nanmedian(self.rewards)) / (np.nanstd(self.rewards) if len(self.rewards) > 1 else 1.0)
        self.rewards.append(r)
        self.t += output.ys.shape[1] * self.dt
        info = self._get_info()
        self.prev_mean = np.nan_to_num(np.nanmean(self.obs, axis=1), nan=0.0, copy=False)
        return (self.prev_mean,
                reward,
                self.t >= 12500,  # 25500 + (4000 * 5),
                np.any(np.isnan(self.obs)),
                info)

    def _is_not_memory(self, e, response, response_reg):
        if e is None:
            return False
        elif (self.exp == "habit" and self.response_reg == 1) or (self.exp == "sens" and self.response_reg == 2):
            return np.mean(e[response, :]) > (self.prev_mean[self.r] / 1.5)
        elif (self.exp == "sens" and self.response_reg == 1) or (self.exp == "habit" and self.response_reg == 2):
            return np.mean(e[response, :]) < (self.prev_mean[self.r] * 1.5)
        elif response_reg == 1:
            return np.mean(e[response, :]) < self.r_scale * self.mean_relax
        return np.mean(e[response, :]) > self.mean_relax / self.r_scale

    def _get_info(self):
        return {"full_obs": self.obs, "is_broken": self._is_not_memory(e=self.obs,
                                                                       response=self.r,
                                                                       response_reg=self.response_reg)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.x = self.get_init_conditions()
        self.t = 10000  # 25500
        self.w = self.mem_data[1]
        self.c = self.mem_data[2]
        self.prev_mean = self.mem_data[-1]
        return self.x, self._get_info()

    def render(self):
        return

    def close(self):
        return


if __name__ == "__main__":
    seed = sys.argv[1]
    env = pickle.load(open(os.path.join("envs", ".".join([str(seed), "pickle"])), "rb"))
    with open(os.path.join("pop", sys.argv[4], ".".join([sys.argv[2], "json"])), "w") as f:
        d = env.step(action={int(i.split(",")[0]): float(i.split(",")[1]) for i in sys.argv[3].split("/")})
        json.dump({"r": d[1].item(),
                   "terminated": bool(d[2]),
                   "truncated": bool(d[3]),
                   "is_broken": bool(d[-1]["is_broken"])}, f)
