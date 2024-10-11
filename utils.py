import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="motion")
    parser.add_argument("--algorithm", type=str, default="rppo")
    parser.add_argument("--policy", type=str, default="MlpLstmPolicy")
    parser.add_argument("--render", type=bool, default=False)
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def get_file_name(seed, task, algorithm, policy):
    return ".".join([str(seed), str(task), str(algorithm), str(policy)])
