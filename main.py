import os
import pickle
import sys
from multiprocessing import Pool

import jax
import numpy as np

from grn import GeneRegulatoryNetwork

if __name__ == "__main__":
    # files = [file for file in os.listdir("memories") if file.endswith("pickle")]
    ids = [3, 4, 10, 69, 16, 22, 23, 26, 27, 29, 31, 631, 204, 50, 37]
    # circuits = []
    # for biomodel_idx in ids:
    #     files = [file for file in os.listdir("memories") if file.startswith(str(biomodel_idx) + ".")]
    #     circuits.extend(np.random.choice(files, size=min(2, len(files))))
    # pickle.dump(circuits, open("circuits.pickle", "wb"))
    # for biomodel_idx in ids:
    for circuit in pickle.load(open("circuits.pickle", "rb")):
        # os.system("python -W ignore al.py --task={}".format(biomodel_idx))
        # for circuit in [file for file in os.listdir("memories") if file.startswith(".".join([str(biomodel_idx), ""]))]:
        os.system(
            "python -W ignore train.py --task={0} --algorithm=es-unif --render=False".format(circuit.strip(".pickle")
                                                                                             .replace(".", "-")))
        print(circuit)
    os.system("python -W ignore plotting.py")
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
