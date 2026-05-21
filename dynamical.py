import nolds
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler
import jax.random as jrandom

from grn4optima import GeneRegulatoryNetwork
from utils import set_seed

DYNAMICAL_PROPERTIES = ["se",
                        "cd.mean",
                        "cd.std",
                        "lyapunov.mean",
                        "lyapunov.std",
                        "lyapunov.max",
                        "dfa",
                        "ghe"]


def get_dynamical_system_features(d):
    se = nolds.sampen(d)
    cds, les = [], []
    for series in d.T:
        try:
            cds.append(nolds.corr_dim(series, emb_dim=2))
        except:
            cds.append(0.0)
        les.append(np.max(nolds.lyap_e(series.astype(np.float32))))
    dfa = nolds.dfa(d)
    ghe = nolds.mfhurst_b(d)[0]
    return [se,
            np.mean(cds),
            np.std(cds),
            np.mean(les),
            np.std(les),
            np.max(les),
            dfa,
            ghe]


def get_dynamical_properties(k, grn, biomodel_id, window=100, num_steps=250000):
    model_data = grn(key=k)[0]
    data = np.zeros((num_steps // window, model_data.ys.shape[0]))
    for col in range(model_data.ys.T[:num_steps].shape[1]):
        data[:, col] = np.nan_to_num(uniform_filter1d(model_data.ys[col, :num_steps].flatten(),
                                                      size=window)[::window],
                                     copy=False)
    data = StandardScaler().fit_transform(data)
    features = get_dynamical_system_features(d=data)
    return features


if __name__ == "__main__":
    set_seed(0)
    with open("dynamics.txt", "w") as file:
        file.write(";".join(DYNAMICAL_PROPERTIES) + "\n")
        key = jrandom.PRNGKey(0)
        for model_id in [2, 3, 4, 5, 6, 10, 69, 16, 17, 22, 21, 23, 26, 27, 483, 29, 31, 631, 203, 204, 209, 210, 275,
                         39, 50, 35, 36, 37, 38]:
            g = GeneRegulatoryNetwork.create(biomodel_idx=model_id)
            g.set_time(10000)
            feats = get_dynamical_properties(k=key, biomodel_id=model_id, grn=g)
            print(model_id)
            file.write(";".join([str(f) for f in feats]) + "\n")
