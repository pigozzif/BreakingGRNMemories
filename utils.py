import argparse
import importlib
import os
import pickle
import random

import gymnasium
import numpy as np
import torch
from autodiscjax.modules.grnwrappers import GRNRollout
from scipy.integrate import ode
from stable_baselines3.common.callbacks import CheckpointCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--task", type=str, default="16-1-2-ass", help="specific memory to test")
    parser.add_argument("--algorithm", type=str, default="es", help="optimization algorithm to train")
    parser.add_argument("--policy", type=str, default="MlpLstmPolicy", help="controller policy (only for RL)")
    parser.add_argument("--exp", type=str, default="ass", help="memory experiment (only ass for associative is available)")
    parser.add_argument("--np", type=int, default=7, help="n. of parallel workers")
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_file_name(seed, task, algorithm, policy):
    return ".".join([str(seed), str(task), str(algorithm), str(policy)])


def get_memory_file(biomodel_idx, r, idx):
    return ".".join([str(biomodel_idx), str(r), str(idx), "pickle"])


def create_system_rollout_module(system_rollout_config, y0=None, w0=None, c=None, t0=None):
    if system_rollout_config.system_type == "grn":
        spec = importlib.util.spec_from_file_location("JaxBioModelSpec", system_rollout_config.model_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        grnstep_cls = getattr(module, "ModelStep")
        grnstep = grnstep_cls(atol=system_rollout_config.atol,
                              rtol=system_rollout_config.rtol,
                              mxstep=system_rollout_config.mxstep)
        if y0 is None:
            y0 = getattr(module, "y0")
        if w0 is None:
            w0 = getattr(module, "w0")
        if c is None:
            c = getattr(module, "c")
        if t0 is None:
            t0 = getattr(module, "t0")
        system_rollout = GRNRollout(n_steps=system_rollout_config.n_system_steps, y0=y0, w0=w0, c=c, t0=t0,
                                    deltaT=system_rollout_config.deltaT, grn_step=grnstep)
    else:
        raise ValueError
    return system_rollout


def discrete2continuous(action, env, num_steps):
    control = np.zeros((num_steps, len(env.get_action_names())))
    start = 0
    window = (env.grn.config.n_secs // (env.grn.NUM_PULSES * 2)) * int(1 / env.dt)
    for pulse in range(env.grn.NUM_PULSES):
        control[start: start + window, action // 2] = -1.0 if action % 2 else 1.0
        start += window * 2
    return control


class SeedEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        return self.env.step(action)


def ode_call_optim(jax_model, y, w, c, t, deltaT):
    solver = ode(jax_model.ratefunc)
    solver.set_integrator('dopri5')
    solver.set_initial_value(y, t)
    while solver.successful() and solver.t < t + deltaT:
        solver.integrate(solver.t + deltaT)
        y[:] = solver.y
    t_new = t + deltaT
    w_new = jax_model.assignmentfunc(y, w, c, t_new)
    return y, w_new, c, t_new


class CheckpointAndRNGCallback(CheckpointCallback):

    def __init__(self, save_freq, name_prefix, save_path="models", save_replay_buffer=True,
                 save_vecnormalize=True, save_rng=True, verbose=0):
        super().__init__(save_freq, save_path, name_prefix, save_replay_buffer, save_vecnormalize, verbose)
        self.save_rng = save_rng
        self.prev_paths = None

    def _checkpoint_path(self, checkpoint_type="", extension=""):
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self):
        if self.prev_paths is not None and self.n_calls % self.save_freq == 0:
            print("SAVEEEEEEEEEE")
            os.remove(self.prev_paths[0])
            os.remove(self.prev_paths[1])
        super()._on_step()
        if self.save_rng and self.n_calls % self.save_freq == 0:
            rng_path = self._checkpoint_path("rng_", extension="pkl")
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            }
            with open(rng_path, "wb") as f:
                pickle.dump(rng_state, f)
            if self.verbose > 1:
                print(f"Saving rng checkpoint to {rng_path}")
            self.prev_paths = (self._checkpoint_path(extension="zip"), rng_path)
        return True
