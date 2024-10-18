import argparse
import importlib
import random

import numpy as np
import torch
from autodiscjax.modules.grnwrappers import GRNRollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="39-0-2-4")
    parser.add_argument("--algorithm", type=str, default="rppo")
    parser.add_argument("--policy", type=str, default="MlpLstmPolicy")
    parser.add_argument("--render", type=bool, default=True)
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def get_file_name(seed, task, algorithm, policy):
    return ".".join([str(seed), str(task), str(algorithm), str(policy)])


def get_memory_file(biomodel_idx, r, ucs, cs):
    return ".".join([str(biomodel_idx), str(r), str(ucs), str(cs), "pickle"])


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
