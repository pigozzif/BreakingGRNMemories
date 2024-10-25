import os
import pickle
from dataclasses import dataclass
from enum import IntEnum

# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import pearsonr, spearmanr, kendalltau

from grn import *
# from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, \
#     global_signal_regression, remove_autocorrelation, corrected_zscore
from utils import parse_args, set_seed


class Regulation(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = 2


@dataclass
class MemoryCircuit(object):
    stimulus: int
    response: int
    stimulus_reg: Regulation
    response_reg: Regulation
    is_ucs: bool


class AssociativeLearning(object):
    NUM_PULSES = 5

    def __init__(self, seed, model_id, us_scale_up=100.0, r_scale_up=2.0, n_secs=2500, **kwargs):
        self.i = model_id
        self.random_key = jrandom.PRNGKey(seed)
        self.grn = GeneRegulatoryNetwork.create(biomodel_idx=model_id, **kwargs)
        self.us_scale_up = us_scale_up
        self.r_scale_up = r_scale_up
        self.n_secs = n_secs
        self.grn.set_time(n_secs=n_secs * 3)
        self.reference = self.relax()
        self.grn.set_time(n_secs=self.n_secs)
        self.relax_t = int(self.n_secs / self.grn.config.deltaT)
        self.relax_y = self.reference.ys[:, :self.relax_t]
        self.genes_ss = self.relax_y[:, -1]
        self.w_ss = self.reference.ws[:, :self.relax_t][:, -1]
        self.bounds = self._get_bounds()
        self.mem_circuits = {}

    def _get_bounds(self):
        bounds = np.zeros((len(self.relax_y), 2))
        bounds[:, 0] = np.min(self.relax_y, axis=1) / self.us_scale_up
        bounds[:, 1] = np.max(self.relax_y, axis=1) * self.us_scale_up
        return bounds

    def relax(self, y0=None, w0=None):
        return self.grn(key=self.random_key, y0=y0, w0=w0)[0]

    def pretest(self):
        curr_circuits = []
        for response in range(len(self.relax_y)):
            for stimulus in range(len(self.relax_y)):
                if response == stimulus:
                    continue
                for regulation in [Regulation(1), Regulation(2)]:
                    curr_circuits.append(self.pretest_for_r(response, stimulus, regulation))
            self.mem_circuits[response] = list(curr_circuits)
            curr_circuits.clear()

    def pretest_for_r(self, response, stimulus, regulation):
        x2 = self.grn.stimulate(key=self.random_key,
                                y0=self.genes_ss,
                                w0=self.w_ss,
                                t0=2500,
                                stimulus={stimulus: self.bounds[stimulus, int(regulation) % 2]},
                                regulation=[regulation])
        if np.mean(x2.ys[response, :]) >= self.r_scale_up * np.mean(self.relax_y[response, :]) and np.mean(
                x2.ys[response, :]) >= self.r_scale_up * np.mean(
            self.reference.ys[response, self.relax_t:self.relax_t * 2]):
            return MemoryCircuit(stimulus=stimulus,
                                 response=response,
                                 stimulus_reg=regulation,
                                 response_reg=Regulation(1),
                                 is_ucs=True)
        elif np.mean(x2.ys[response, :]) <= (1 / self.r_scale_up) * np.mean(self.relax_y[response, :]) and np.mean(
                x2.ys[response, :]) <= (1 / self.r_scale_up) * np.mean(
            self.reference.ys[response, self.relax_t:self.relax_t * 2]):
            return MemoryCircuit(stimulus=stimulus,
                                 response=response,
                                 stimulus_reg=regulation,
                                 response_reg=Regulation(2),
                                 is_ucs=True)
        return MemoryCircuit(stimulus=stimulus,
                             response=response,
                             stimulus_reg=regulation,
                             response_reg=Regulation(0),
                             is_ucs=False)

    def eval_mem_for_r(self, response):
        if not self.mem_circuits[response]:
            return
        cs_list = [circuit for circuit in self.mem_circuits[response] if not circuit.is_ucs]
        for ucs_circuit in [circuit for circuit in self.mem_circuits[response] if circuit.is_ucs]:
            self.test_associative_memory(ucs_circuit, cs_list)

    def test_associative_memory(self, ucs_circuit, cs_list):
        is_mem = False
        for cs_circuit in cs_list:
            if ucs_circuit.stimulus == cs_circuit.stimulus:
                continue
            e1 = self.grn.stimulate(key=self.random_key,
                                    y0=self.genes_ss,
                                    w0=self.w_ss,
                                    t0=self.n_secs,
                                    stimulus={ucs_circuit.stimulus:
                                                  self.bounds[ucs_circuit.stimulus, int(ucs_circuit.stimulus_reg) % 2],
                                              cs_circuit.stimulus:
                                                  self.bounds[cs_circuit.stimulus, int(cs_circuit.stimulus_reg) % 2]},
                                    regulation=[ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
            up_down_r = self.is_r_regulated(e1, cs_circuit.response)
            if int(up_down_r) != 0:
                e2 = self.grn.stimulate(key=self.random_key,
                                        y0=e1.ys[:, -1],
                                        w0=e1.ws[:, -1],
                                        t0=self.n_secs * 2,
                                        stimulus={},
                                        regulation=[])
                e3 = self.grn.stimulate(key=self.random_key,
                                        y0=e2.ys[:, -1],
                                        w0=e2.ws[:, -1],
                                        t0=self.n_secs * 3,
                                        stimulus={cs_circuit.stimulus:
                                                      self.bounds[
                                                          cs_circuit.stimulus, int(cs_circuit.stimulus_reg) % 2]},
                                        regulation=[cs_circuit.stimulus_reg])
                is_mem = self.is_memory(e3, ucs_circuit.response, up_down_r)
                if is_mem:
                    # try:
                    #     with open("activity.txt", "a") as file:
                    #         info = self.compute_circuit_info(data=np.nan_to_num(e3.ys))["emergence"]
                    #         info = np.nan_to_num(info,
                    #                              neginf=np.nanmin(info[info != -np.inf]),
                    #                              posinf=np.nanmax(info[info != np.inf]))
                    #         traj = np.diff(StandardScaler().fit_transform(e3.ys))
                    #         traj = np.nan_to_num(traj,
                    #                              neginf=np.nanmin(traj[traj != -np.inf]),
                    #                              posinf=np.nanmax(traj[traj != np.inf]))
                    #         traj = np.mean(traj, axis=0)[-len(info):]
                    #         file.write(";".join([str(self.i),
                    #                              str(cs_circuit.response),
                    #                              str(max([int(file.split(".")[2]) for file in os.listdir("memories")
                    #                                       if file.startswith(
                    #                                      ".".join([str(self.i), str(cs_circuit.response), ""]))] + [0])
                    #                                  + 1),
                    #                              str(pearsonr(info, traj).statistic),
                    #                              str(spearmanr(info, traj).statistic),
                    #                              str(kendalltau(info, traj).statistic)]) + "\n")
                    # except:
                    #     pass
                    self.save_memory(e3,
                                     r=cs_circuit.response,
                                     ucs=ucs_circuit.stimulus,
                                     cs=cs_circuit.stimulus,
                                     response_reg=up_down_r,
                                     stimulus_reg=cs_circuit.stimulus_reg)
                del e2, e3
            del e1
        return is_mem

    # def compute_circuit_info(self, data):
    #     data = corrected_zscore(np.array(data), axis=1)
    #     data = global_signal_regression(data)
    #     data = remove_autocorrelation(data)
    #     data = data.astype(np.float64, copy=False)
    #     info = {}
    #     mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    #     mib = minimum_information_bipartition(mi_mat, noise=True)
    #     component_1 = data[mib[0], :].mean(axis=0)
    #     component_2 = data[mib[1], :].mean(axis=0)
    #     data_reduced = np.vstack((component_1, component_2))
    #     phi_results = local_phi_id(0, 1, data_reduced)
    #     info["synergy"] = phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"]
    #    info["causation"] = phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))][
    #         "pi"]
    #     info["emergence"] = np.nansum([info["synergy"], info["causation"]], axis=0)
    #     return info

    def is_r_regulated(self, e1, response):
        return self.is_r_regulated_gen(x1=self.relax_y[response, :],
                                       e1=e1,
                                       ref=self.reference.ys[response, self.relax_t:self.relax_t * 2],
                                       response=response)

    def is_r_regulated_gen(self, x1, e1, ref, response):
        mean_e1 = np.mean(e1.ys[response, :])
        if mean_e1 >= self.r_scale_up * np.mean(x1) \
                and mean_e1 >= self.r_scale_up * np.mean(ref):
            return Regulation(1)
        elif mean_e1 <= (1 / self.r_scale_up) * np.mean(x1) \
                and mean_e1 <= (1 / self.r_scale_up) * np.mean(ref):
            return Regulation(2)
        return Regulation(0)

    def is_memory(self, e2, response, response_reg):
        mean_relax = np.mean(self.relax_y[response, :])
        if response_reg == 1:
            return np.mean(e2.ys[response, :]) >= self.r_scale_up * mean_relax
        return np.mean(e2.ys[response, :]) <= mean_relax / self.r_scale_up

    def save_memory(self, e3, r, ucs, cs, response_reg, stimulus_reg):
        new_bounds = self.bounds.copy()
        # new_bounds[:, 0] *= self.us_scale_up
        # new_bounds[:, 1] /= self.us_scale_up
        idx = max([int(file.split(".")[2]) for file in os.listdir("memories")
                   if file.startswith(".".join([str(self.i), str(r), ""]))] + [0]) + 1
        pickle.dump([e3.ys[:, -1],
                     e3.ws[:, -1],
                     e3.cs[:, -1],
                     new_bounds,
                     np.mean(self.relax_y[r, :]),
                     int(response_reg),
                     int(stimulus_reg),
                     self.r_scale_up,
                     r,
                     ucs,
                     cs],
                    open(os.path.join("memories",
                                      ".".join([str(self.i),
                                                str(r),
                                                str(idx),
                                                "pickle"])),
                         "wb"))


def learn(seed, i):
    al = AssociativeLearning(seed=seed, model_id=i)
    al.pretest()
    for r in al.mem_circuits.keys():
        al.eval_mem_for_r(response=r)


if __name__ == "__main__":
    # 26, 27, 29, 31
    arguments = parse_args()
    set_seed(arguments.seed)
    learn(arguments.seed, arguments.task.split("-")[0])
