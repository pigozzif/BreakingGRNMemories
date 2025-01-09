import os
import pickle
import sys
from multiprocessing import Pool

import jax
import numpy as np

from grn import GeneRegulatoryNetwork

# TODO: remake associative al.py exps until 69 including
# TODO: 21-habit
if __name__ == "__main__":
    # files = [file for file in os.listdir("memories") if file.endswith("pickle")]
    ids = [2, 3, 4, 5, 6, 10, 69, 16, 17, 22, 21, 23, 26, 27, 483, 29, 31, 631, 203, 204, 209, 210, 275, 39, 50]
    with Pool(len(ids)) as pool:
        pool.map(os.system, ["python3.10 -W ignore al.py --task={0} --exp=habit".format(i) for i in ids])
    exit()
    # circuits = []
    # for biomodel_idx in ids:
    #     files = [file for file in os.listdir("memories/habit") if file.startswith(str(biomodel_idx) + ".")]
    #     circuits.extend(np.random.choice(files, size=min(2, len(files))))
    # pickle.dump(circuits, open("circuits-habit.pickle", "wb"))
    # for biomodel_idx in ids:
    # for circuit in pickle.load(open("circuits.pickle", "rb")):
    exp = sys.argv[2]
    # for circuit in pickle.load(open("circuits-{}.pickle".format(exp), "rb")):
    for circuit in os.listdir("memories/habit"):
        if not circuit.startswith("4."):
            continue
        # os.system("python -W ignore al.py --task={} --exp=habit".format(biomodel_idx))
        # for circuit in [file for file in os.listdir("memories/sens") if file.startswith(".".join([str(biomodel_idx), ""]))]:
        os.system(
            "python -W ignore train.py --seed={0} --task={1} --exp={2} --algorithm=rppo --render=False".format(
                sys.argv[1],
                circuit.strip(
                    ".pickle")
                .replace(".",
                         "-") + "-" + exp,
                exp))
    # with Pool(7) as pool:
    #     pool.map(os.system,
    #              ["python -W ignore sgd.py --seed={0} --task={1} --algorithm=sgd --render=False".format(sys.argv[1],
    #                                                                                                     circuit.strip(
    #                                                                                                         ".pickle")
    #                                                                                                     .replace(".",
    #                                                                                                              "-") + "-" + exp)
    #               for circuit in pickle.load(open("circuits-{}.pickle".format(exp), "rb"))])
    # print(biomodel_idx)
    # os.system("python -W ignore plotting.py")
    # key = jax.random.PRNGKey(0)
    # np.random.seed(0)
    # for biomodel_idx in ids:
    #     grn = GeneRegulatoryNetwork.create(biomodel_idx=biomodel_idx,
    #                                        deltaT=0.01,
    #                                        n_secs=10000)
    #     output, data = grn(key=key)
    #     np.save("../GRNs/gene_trajectories/{}.npy".format(biomodel_idx), output.ys)
    # exit()
#    with Pool(int(sys.argv[2])) as pool:
#        pool.map(os.system, ["python -W ignore train.py --task={1} --algorithm=single --render=False"
#                 .format(sys.argv[1],
#                         model_idx.replace(".", "-")) for model_idx in pickle.load(open("circuits.pickle", "rb"))])
