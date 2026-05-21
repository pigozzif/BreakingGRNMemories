import os
import random

import jax
import numpy as np
import pandas as pd
from gym.spaces import Discrete

from al import AssociativeLearning, MemoryCircuit, Regulation
from grn import GeneRegulatoryNetwork
from information import preprocess_data, local_phi_id, local_phi_r, minimum_information_bipartition, \
    mutual_information_matrix
from train import get_env


PHASES = ["relax", "train", "relapse", "test", "reset", "verify"]
MEASURES = ["emergence"]


def test_network(al, ucs_circuit, cs_circuit, env, control, action):
    e1 = al.grn.stimulate(key=al.random_key,
                          y0=al.genes_ss,
                          w0=al.w_ss,
                          t0=al.n_secs,
                          stimulus={ucs_circuit.stimulus:
                                        al.bounds[ucs_circuit.stimulus, int(ucs_circuit.stimulus_reg) % 2],
                                    cs_circuit.stimulus:
                                        al.bounds[cs_circuit.stimulus, int(cs_circuit.stimulus_reg) % 2]})
    up_down_r = al.is_r_regulated(e1, cs_circuit.response)
    # if int(up_down_r) != 0:
    e2 = al.relax(y0=e1.ys[:, -1],
                  w0=e1.ws[:, -1],
                  t0=al.n_secs * 2)
    e3 = al.grn.stimulate(key=al.random_key,
                          y0=e2.ys[:, -1],
                          w0=e2.ws[:, -1],
                          t0=al.n_secs * 3,
                          stimulus={ucs_circuit.stimulus:
                                        al.bounds[
                                            ucs_circuit.stimulus, int(ucs_circuit.stimulus_reg) % 2]})
    is_mem = al.is_memory(e3, ucs_circuit.response, up_down_r)
    stimuli = {control: env.mem_data[3][control][action % 2]}
    e4 = al.grn.stimulate(key=al.random_key,
                          y0=e3.ys[:, -1],
                          w0=e3.ws[:, -1],
                          t0=al.n_secs * 4,
                          stimulus=stimuli)
    return is_mem, up_down_r, np.hstack([al.relax_y, e1.ys, e2.ys, e3.ys, e4.ys])


def compute_integrated_info(data):
    # print(np.any(np.isnan(data)))
    data = preprocess_data(data)
    # print(np.any(np.isnan(data)))
    data = np.nan_to_num(data, nan=0.0, copy=True)
    # print(np.any(np.isnan(data)))
    data = data.astype(np.float64, copy=False)
    info = {}
    mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    mib = minimum_information_bipartition(mi_mat, noise=True)
    component_1 = data[mib[0], :].mean(axis=0)
    component_2 = data[mib[1], :].mean(axis=0)
    data_reduced = np.vstack((component_1, component_2))
    phi_results = local_phi_id(0, 1, data_reduced)
    info["synergy"] = np.nan_to_num(phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"], nan=0.0, posinf=0.0, neginf=0.0)
    info["causation"] = np.nan_to_num(phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))]["pi"], nan=0.0, posinf=0.0, neginf=0.0)
    # info["integrated"] = np.nan_to_num(local_phi_r(phi_results), nan=0.0, posinf=0.0, neginf=0.0)
    info["emergence"] = info["synergy"] + info["causation"]
    # period = 250000
    # for start, period_name in zip(range(0, period * len(PHASES), period), PHASES):
    #     for traj in info.values():
    #         print(np.median(traj[start: start + period]), traj[start: start + period].shape)
    return info


if __name__ == "__main__":
    num = 0
    total = 0
    is_random = False
    key = jax.random.PRNGKey(0)
    col_names = ["model_id", "response_id", "cs_id", "ucs_id", "control", "reg", "up_down_r", "is_mem"]
    for phase in PHASES:
        for measure in MEASURES:
            col_names.append(".".join([phase, measure]))
    file_name = "info.txt" if not is_random else "info_random.txt"
    with open(file_name, "w") as file:
        file.write(";".join(col_names) + "\n")

    sim_data = {}
    for file in os.listdir("output"):
        if "single" not in file or not file.endswith("csv"):
            continue
        result = pd.read_csv(os.path.join("output", file), sep=",", skiprows=1)
        if not result["is_broken"].any():
            continue
        total += 1
        for exp in ["ass"]:
            try:
                env = get_env(env_name="-".join([file.split(".")[1], exp]), seed=0)
                break
            except FileNotFoundError as e:
                continue
        else:
            continue
        grn_idx = int(file.split(".")[1].split("-")[0])
        env.action_space = Discrete(n=(env.obs_dim - 2) * 2, seed=0)
        for i, action in enumerate(range(env.action_space.n)):
            is_broken = result.loc[i, "is_broken"]
            if (is_broken and not is_random) or (not is_broken and is_random):
                control = env.action_map[action // 2]
                num += 1
                if grn_idx not in sim_data:
                    sim_data[grn_idx] = [(env, control, action)]
                else:
                    sim_data[grn_idx].append((env, control, action))

    for idx, tasks in sim_data.items():
        print(idx)
        al = AssociativeLearning(seed=0, model_id=idx)
        tasks = random.sample(tasks, k=min(len(tasks), 10))
        for (env, control, action) in tasks:
            for reg in [Regulation(1), Regulation(2)]:
                ucs = MemoryCircuit(stimulus=env.mem_data[9],
                                    stimulus_reg=reg,
                                    response=env.r,
                                    response_reg=env.response_reg,
                                    is_ucs=True,
                                    ys=None,
                                    ws=None,
                                    cs=None)
                cs = MemoryCircuit(stimulus=env.s,
                                   stimulus_reg=env.stimulus_reg,
                                   response=env.r,
                                   response_reg=env.response_reg,
                                   is_ucs=False,
                                   ys=None,
                                   ws=None,
                                   cs=None)
                mem, udr, d = test_network(al=al, ucs_circuit=ucs, cs_circuit=cs, env=env, control=control,
                                           action=action)
                information = compute_integrated_info(data=d)
                with open(file_name, "a") as f:
                    measures = [idx, ucs.response, ucs.stimulus, cs.stimulus, control, reg, udr, mem]
                    period = int(al.grn.config.n_secs / al.grn.config.deltaT)
                    for start in range(0, period * len(PHASES), period):
                        for measure in MEASURES:
                            measures.append(np.median(information[measure][start: start + period]))
                    f.write(";".join([str(measure) for measure in measures]) + "\n")
    print(num)
