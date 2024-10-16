import pickle
from dataclasses import dataclass
from enum import IntEnum

import jax.numpy as jnp
from autodiscjax import DictTree
from autodiscjax.modules import grnwrappers

from grn import *
from utils import parse_args, set_seed, get_memory_file


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

    def stimulate(self, y0, w0, stimulus, regulation):
        if not isinstance(stimulus, list):
            stimulus = [stimulus]
        if not isinstance(regulation, list):
            regulation = [regulation]
        intervention_params = DictTree()
        for s, regulation in zip(stimulus, regulation):
            intervention_params.y[s] = jnp.array([self.bounds[s, int(regulation) % 2]])
        intervals = []
        window = self.grn.config.n_secs // (self.NUM_PULSES * 2)
        for _ in stimulus:
            start = 0
            for pulse in range(self.NUM_PULSES):
                intervals.append([start, start + window])
                start += window * 2
        intervention_fn = grnwrappers.PiecewiseSetConstantIntervention(
            time_to_interval_fn=grnwrappers.TimeToInterval(
                intervals=intervals))
        return self.grn(key=self.random_key,
                        y0=y0,
                        w0=w0,
                        intervention_fn=intervention_fn,
                        intervention_params=intervention_params)[0]

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
        x2 = self.stimulate(self.genes_ss, self.w_ss, stimulus, regulation)
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
        us, pairing, transfer, associative, consolidation = [], [], [], [], []
        if not self.mem_circuits[response]:
            return us, pairing, transfer, associative, consolidation
        cs_list = [circuit for circuit in self.mem_circuits[response] if not circuit.is_ucs]
        for ucs_circuit in [circuit for circuit in self.mem_circuits[response] if circuit.is_ucs]:
            self.test_associative_memory(ucs_circuit, cs_list)
        return us, pairing, transfer, associative, consolidation

    def test_associative_memory(self, ucs_circuit, cs_list):
        is_mem = False
        for cs_circuit in cs_list:
            if ucs_circuit.stimulus == cs_circuit.stimulus:
                continue
            e1 = self.stimulate(self.genes_ss, self.w_ss, [ucs_circuit.stimulus, cs_circuit.stimulus],
                                [ucs_circuit.stimulus_reg, cs_circuit.stimulus_reg])
            up_down_r = self.is_r_regulated(e1, cs_circuit.response)
            if int(up_down_r) != 0:
                e2 = self.stimulate(e1.ys[:, -1], e1.ws[:, -1], [], [])
                e3 = self.stimulate(e2.ys[:, -1], e2.ws[:, -1], cs_circuit.stimulus, cs_circuit.stimulus_reg)
                is_mem = self.is_memory(e1, e3, ucs_circuit.response, up_down_r)
                if is_mem:
                    self.save_memory(e3, r=cs_circuit.response, ucs=ucs_circuit.stimulus, cs=cs_circuit.stimulus)
                del e2, e3
            del e1
        return is_mem

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

    def is_memory(self, e1, e2, response, response_reg):
        mean_relax = np.mean(self.relax_y[response, :])
        if response_reg == 1:
            return np.mean(e2.ys[response, :]) >= mean_relax + (
                    np.mean(e1.ys[response, :]) - mean_relax) / 2.0
        return np.mean(e2.ys[response, :]) <= mean_relax - (
                mean_relax - np.mean(e1.ys[response, :])) / 2.0

    def save_memory(self, e3, r, ucs, cs):
        new_bounds = self.bounds.copy()
        new_bounds[:, 0] *= self.us_scale_up
        new_bounds[:, 1] /= self.us_scale_up
        pickle.dump([e3.ys[:, -1], e3.ws[:, -1], e3.cs[:, -1], new_bounds],
                    open(os.path.join("memories",
                                      get_memory_file(biomodel_idx=self.i,
                                                      r=str(r),
                                                      ucs=str(ucs),
                                                      cs=str(cs))),
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
