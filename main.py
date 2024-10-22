import os
from multiprocessing import Pool

import numpy as np

if __name__ == "__main__":
    # files = [file for file in os.listdir("memories") if file.endswith("pickle")]
    # files = np.random.choice(files, size=50, replace=False)
    # for file in files:
    for biomodel_idx in [2, 5, 6, 17, 483, 203, 275, 36, 38]:
        os.system("python -W ignore al.py --task={}".format(biomodel_idx))
        # os.system("python -W ignore train.py --task={} --algorithm=exhaustive-multiple --render=False".format("-".join(file.split(".")[:-1])))
        for file in [f for f in os.listdir("memories") if f.endswith("pickle") and f.startswith(str(biomodel_idx) + ".")]:
            os.system("python -W ignore train.py --task={} --algorithm=exhaustive-single --render=False".format("-".join(file.split(".")[:-1])))
    # with Pool(4) as pool:
    #     pool.map(os.system, ["python -W ignore train.py --task={}".format("-".join(file.split(".")[:-1]))
    #                          for file in os.listdir("memories") if file.endswith("pickle")])
