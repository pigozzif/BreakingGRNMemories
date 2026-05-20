from multiprocessing import Pool

import numpy as np

from al import AssociativeLearning, Regulation
from utils import parse_args, set_seed


def is_ass_mem(x, e, response, response_reg, al):
    mean_relax = np.mean(x[response, :])
    if response_reg == 1:
        return np.mean(e[response, :]) >= al.r_scale_up * mean_relax
    return np.mean(e[response, :]) <= mean_relax / al.r_scale_up


def learn(args):
    seed, i = args
    al = AssociativeLearning(seed=seed, model_id=i, n_secs=10000)
    mem_circuits = {}
    x = al.reference
    num = 0
    for response in range(len(x.ys)):
        for regulation in [Regulation(1), Regulation(2)]:
            if is_ass_mem(x=x.ys[:, :2500],
                          e=x.ys[:, 7500:],
                          response_reg=regulation,
                          response=response,
                          al=al):
                num += 1
        if num > 0:
            mem_circuits[response] = num
        num = 0
    with open("quiescent.txt", "a") as f:
        f.write(";".join([str(i),
                          str(sum([v for v in mem_circuits.values()])),
                          str(len(mem_circuits))]) + '\n')


if __name__ == "__main__":
    # 26, 27, 29, 31
    arguments = parse_args()
    set_seed(arguments.seed)
    with open("quiescent.txt", "w") as file:
        file.write(";".join(["i", "num.memories", "num.responses"]) + '\n')
    with Pool(5) as pool:
        # pool.map(learn, [(arguments.seed, idx) for idx in [3, 4, 10, 16, 22, 23, 26, 27, 29, 31, 37, 50, 69, 204, 631]])
        pool.map(learn, [(arguments.seed, idx) for idx in [2, 3, 4, 5, 6, 10, 69, 16, 17, 22, 21, 23, 26, 27, 483,
                                                           29, 31, 631, 203, 204, 209, 210, 275, 39, 50, 35, 36, 37,
                                                           38]])
