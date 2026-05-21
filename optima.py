from multiprocessing import Pool
import jax.random as jrandom
import numpy as np

from al4optima import AssociativeLearning, get_num_no_mem, learn
from dynamical import get_dynamical_properties, DYNAMICAL_PROPERTIES
from information import preprocess_data, compute_circuit_info
from network import get_network_properties, NETWORK_PROPERTIES


def simulate_random(args):
    try:
        seed, biomodel_idx = args
        key = jrandom.PRNGKey(seed)
        np.random.seed(seed)
        al = AssociativeLearning(seed=seed,
                                 model_id=biomodel_idx,
                                 random=True)
        num_ucs_circuits, memories = learn(al, biomodel_idx)
        num_no_mem = get_num_no_mem(num_ucs_circuits=num_ucs_circuits, memories=memories)
        output, _ = al.grn(key=key)
        processed_data = preprocess_data(np.array(output.ys))
        info = compute_circuit_info(data=processed_data)
        network_properties = get_network_properties(biomodel_id=biomodel_idx)
        dynamical_properties = get_dynamical_properties(k=key, grn=al.grn, biomodel_id=biomodel_idx)
    except:
        return None
    print(f"Done: {seed} {biomodel_idx}")
    return [seed, biomodel_idx] + network_properties + dynamical_properties + [1.0 - num_no_mem / (num_ucs_circuits + 1e-6)] + [np.median(info["emergence"])]


if __name__ == "__main__":
    networks = [3, 4, 16, 29, 31, 39]
    all_results = []
    with open("optima.txt", "w") as file:
        file.write(";".join(["seed", "model_id"] + NETWORK_PROPERTIES + DYNAMICAL_PROPERTIES + ["memories.perc"] +
                                                    ["emergence.median"])
                   + "\n")

    for s in range(0, 2):
        with Pool(3) as pool:
            results = pool.map(simulate_random, [(s, network) for network in networks])
            all_results.extend(results)

    with open("optima.txt", "a") as file:
        for result in results:
            if result is not None:
                file.write(";".join([str(r) for r in result]) + "\n")
