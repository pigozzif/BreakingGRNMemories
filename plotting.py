import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from envs import GRNEnv
from grn import GeneRegulatoryNetwork
from utils import create_system_rollout_module


def _flush_plot(name):
    plt.savefig(os.path.join("figures", name))
    plt.close()


def _load_all_data(directory="output", reduce=True):
    data = None
    for file in os.listdir(directory):
        if not file.endswith("csv"):
            continue
        d = pd.read_csv(os.path.join(directory, file), sep=",", skiprows=1)
        d["seed"] = int(file.split(".")[0])
        d["idx"] = file.split(".")[1]
        d["any_broken"] = any(d["is_broken"])
        d["algorithm"] = file.split(".")[2].split("-")[0]
        data = d if data is None else pd.concat([data, d], axis=0)
    if reduce:
        circuits = [c.replace(".pickle", "").replace('.', '-') for c in pickle.load(open("circuits.pickle", "rb"))]
        data = data.query("idx in @circuits")
    return data


def plot_reward(file_name):
    task = file_name.split(".")[-3]
    for file in os.listdir("output"):
        if task not in file:
            continue
        data = pd.read_csv(os.path.join("output", file), sep=",", skiprows=1)
        plt.plot(data["r"][1:], label="-".join([file.split(".")[-3], file.split(".")[-2]]))
    plt.xlabel("RL time steps", fontsize=10)
    plt.ylabel("reward", fontsize=10)
    plt.legend()
    _flush_plot(".".join([task, "png"]))


def plot_broken_memories(names={"single": "single", "es": "# stim.\n+\nstim. val", "ga": "# stim."}):
    data = _load_all_data(reduce=True)
    algorithms = [algorithm for (algorithm,), _ in data.groupby(["algorithm"])]
    plt.bar(np.arange(len(algorithms)),
            height=[np.mean([any(d["is_broken"]) for _, d in traj.groupby(["idx"])]) * 100
                    for _, traj in data.groupby(["algorithm"])],
            # yerr=[np.std([any(d["is_broken"]) for _, d in traj.groupby(["idx"])])
            #       for _, traj in data.groupby(["algorithm"])],
            capsize=5)
    plt.ylabel("% of broken memories", fontsize=15)
    plt.xticks(np.arange(len(algorithms)), [names[a] for a in algorithms], fontsize=10)
    plt.ylim(0.0, 100.0)
    _flush_plot("broken.png")


def plot_distribution_single():
    data = _load_all_data(reduce=False)
    data = data[data["algorithm"] == "single"]
    data["network"] = data.apply(lambda row: row["idx"].split("-")[0], axis=1)
    ids = [i for (i,), traj in data.groupby(["network"])]
    plt.bar(np.arange(len(ids)),
            height=[np.mean([any(d["is_broken"]) for _, d in traj.groupby(["idx"])]) * 100
                    for _, traj in data.groupby(["network"])])
    plt.ylabel("% of broken memories", fontsize=15)
    plt.xticks(np.arange(len(ids)), ids, fontsize=10)
    _flush_plot("single_by_id.png")


def fitness_landscape(idx=(3, 2, 1)):
    grn = GeneRegulatoryNetwork.create(biomodel_idx=idx[0])
    env = GRNEnv(seed=0,
                 grn=grn,
                 obs_dim=len(create_system_rollout_module(grn.config).grn_step.y_indexes),
                 r=idx[1],
                 idx=idx[2])
    bounds = env.mem_data[0]
    b = 1500
    step = 1
    values = []
    actions = []
    for a in range(0, b, step):
        a -= b // 2
        a /= step
        print(a)
        _, reward, *_ = env.step(action={0: bounds[env.action_map[0]] * np.exp(a)})
        values.append(reward)
        actions.append(a)
        env.reset()
    plt.plot(actions, values)
    _flush_plot("landscape_{}.png".format("-".join([str(i) for i in idx])))


if __name__ == "__main__":
    # plot_distribution_single()
    plot_broken_memories()
    # fitness_landscape()
