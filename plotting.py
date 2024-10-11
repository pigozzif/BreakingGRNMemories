import os

import pandas as pd
from matplotlib import pyplot as plt


def plot_reward(file_name):
    task = file_name.split(".")[-3]
    for file in os.listdir("output"):
        if task not in file:
            continue
        data = pd.read_csv(os.path.join("output", file), sep=",", skiprows=1)
        plt.plot(data["r"][1:], label="-".join([file.split(".")[-3], file.split(".")[-2]]))
    plt.xlabel("RL time steps")
    plt.ylabel("reward")
    plt.legend()
    plt.savefig(os.path.join("figures", ".".join([task, "png"])))
    plt.close()
