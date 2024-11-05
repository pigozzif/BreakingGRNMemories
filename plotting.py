import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _flush_plot(name):
    plt.savefig(os.path.join("figures", name))
    plt.close()


def _load_all_data(directory="output"):
    data = None
    for file in os.listdir(directory):
        if not file.endswith("csv"):
            continue
        d = pd.read_csv(os.path.join(directory, file), sep=",", skiprows=1)
        d["seed"] = int(file.split(".")[0])
        d["id"] = file.split(".")[1]
        d["any_broken"] = any(d["is_broken"])
        d["algorithm"] = file.split(".")[2]
        data = d if data is None else pd.concat([data, d], axis=0)
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


def plot_broken_memories():
    data = _load_all_data()
    algorithms = data["algorithm"].unique()
    plt.bar(np.arange(len(algorithms)),
            height=[np.mean([any(d["is_broken"]) for _, d in traj.groupby(["id"])])
                    for _, traj in data.groupby(["algorithm"])],
            yerr=[np.std([any(d["is_broken"]) for _, d in traj.groupby(["id"])])
                  for _, traj in data.groupby(["algorithm"])],
            capsize=5)
    plt.ylabel("% of broken memories", fontsize=15)
    plt.xticks(np.arange(len(algorithms)), algorithms, fontsize=15)
    plt.ylim(0.0, 1.5)
    _flush_plot("broken.png")


if __name__ == "__main__":
    plot_broken_memories()
