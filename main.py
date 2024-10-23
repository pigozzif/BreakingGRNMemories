import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

if __name__ == "__main__":
    # files = [file for file in os.listdir("memories") if file.endswith("pickle")]
    # files = np.random.choice(files, size=50, replace=False)
    # for file in files:
    for biomodel_idx in pickle.load(open("circuits.pickle", "rb")):
        # os.system("python -W ignore al.py --task={}".format(biomodel_idx))
        os.system("python -W ignore train.py -seed={0} --task={1} --algorithm=ga --render=False".format(sys.argv[1],
                                                                                                        biomodel_idx))
    # with Pool(4) as pool:
    #     pool.map(os.system, ["python -W ignore train.py --task={}".format("-".join(file.split(".")[:-1]))
    #                          for file in os.listdir("memories") if file.endswith("pickle")])
