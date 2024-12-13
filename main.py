import os
import pickle
import sys
from multiprocessing import Pool

import jax
import numpy as np

from grn import GeneRegulatoryNetwork
# TODO: remake associative al.py exps until 69 including
if __name__ == "__main__":
    # files = [file for file in os.listdir("memories") if file.endswith("pickle")]
    ids = [2, 3, 4, 5, 6, 10, 69, 16, 17, 22, 21, 23, 26, 27, 483, 29, 31, 631, 203, 204, 209, 210, 275, 39, 50, 35, 36,
           37, 38]
    # circuits = []
    # for biomodel_idx in ids:
    #     files = [file for file in os.listdir("memories") if file.startswith(str(biomodel_idx) + ".")]
    #     circuits.extend(np.random.choice(files, size=min(2, len(files))))
    # pickle.dump(circuits, open("circuits.pickle", "wb"))
    for biomodel_idx in ids:
    # for circuit in pickle.load(open("circuits.pickle", "rb")):
        # os.system("python -W ignore al.py --task={} --exp=habit".format(biomodel_idx))
        for circuit in [file for file in os.listdir("memories/habit") if file.startswith(".".join([str(biomodel_idx), ""]))]:
            os.system(
             "python -W ignore train.py --task={0} --algorithm=single --render=False".format(circuit.strip(".pickle")
                                                                                              .replace(".", "-") + "-habit"))
        print(biomodel_idx)
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
