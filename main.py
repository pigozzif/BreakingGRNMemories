import os
from multiprocessing import Pool

if __name__ == "__main__":
    with Pool(4) as pool:
        pool.map(os.system, ["python -W ignore train.py --task={}".format("-".join(file.split(".")[:-1]))
                             for file in os.listdir("memories") if file.endswith("pickle")])
