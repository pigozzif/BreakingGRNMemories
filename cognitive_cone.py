import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import qmc

from al import AssociativeLearning
from utils import create_system_rollout_module, parse_args, set_seed


def simulate(al, r, ucs, cs, stimulus_reg, init):
    relax = al.grn(key=al.random_key,
                   y0=init)[0]
    e1 = al.grn.stimulate(key=al.random_key,
                          y0=relax.ys[:, -1],
                          w0=relax.ws[:, -1],
                          t0=al.n_secs,
                          stimulus={ucs: al.bounds[ucs, int(stimulus_reg) % 2],
                                    cs: al.bounds[cs, int(stimulus_reg) % 2]})
    up_down_r = al.is_r_regulated(e1, r)
    e2 = al.relax(y0=e1.ys[:, -1],
                  w0=e1.ws[:, -1],
                  t0=al.n_secs * 2)
    e3 = al.grn.stimulate(key=al.random_key,
                          y0=e2.ys[:, -1],
                          w0=e2.ws[:, -1],
                          t0=al.n_secs * 3,
                          stimulus={ucs: al.bounds[ucs, int(stimulus_reg) % 2]})
    is_mem = al.is_memory(e3, r, up_down_r)
    return is_mem and int(up_down_r) != 0, np.hstack([relax.ys, e1.ys, e2.ys, e3.ys])


def compute_cognitive_cone(seed, memory, num_samples=100):
    idx = int(memory.split(".")[0])
    al = AssociativeLearning(seed=seed, model_id=idx)
    mem = pickle.load(open(os.path.join("old_memories", "ass", memory), "rb"))
    sampler = qmc.LatinHypercube(d=len(create_system_rollout_module(al.grn.config).grn_step.y_indexes))
    samples = sampler.random(n=num_samples)
    scaled_samples = qmc.scale(samples, l_bounds=[b[0] for b in al.bounds], u_bounds=[b[1] for b in al.bounds])
    _, base = simulate(al=al,
                       r=int(memory.split(".")[1]),
                       ucs=mem[9],
                       cs=mem[10],
                       stimulus_reg=mem[6],
                       init=None)
    new_data = []
    for i, sample in enumerate(scaled_samples):
        is_mem, data = simulate(al=al,
                                r=int(memory.split(".")[1]),
                                ucs=mem[9],
                                cs=mem[10],
                                stimulus_reg=mem[6],
                                init=sample)
        new_data.append(np.nanmean(np.abs(base - data), axis=0))
        # pickle.dump([is_mem, new_data], open(os.path.join("cone", ".".join([str(i), memory])), "wb"))
    median = np.nanmedian(np.array(new_data), axis=0)
    plt.plot(median)
    err = np.nanstd(np.array(new_data), axis=0)
    plt.fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
    plt.vlines(x=[250000, 500000, 750000], ymin=0.0, ymax=np.max(median), color="red", alpha=0.25)
    plt.savefig("figures/cone.{}.png".format(memory.replace(".pickle", "")))
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)
    compute_cognitive_cone(seed=args.seed,
                           memory=".".join([args.task, "pickle"]).replace("-", "."))
