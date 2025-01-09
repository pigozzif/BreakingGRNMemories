import os
import pickle
from dataclasses import dataclass
from enum import IntEnum

from grn import *
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
    ys: np.array
    ws: np.array


class AssociativeLearning(object):
    NUM_PULSES = 5
    US_SCALE = 100.0

    def __init__(self, seed, model_id, r_scale_up=2.0, n_secs=2500, **kwargs):
        self.i = model_id
        self.random_key = jrandom.PRNGKey(seed)
        self.grn = GeneRegulatoryNetwork.create(biomodel_idx=model_id, **kwargs)
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
        bounds[:, 0] = np.min(self.relax_y, axis=1) / self.US_SCALE
        bounds[:, 1] = np.max(self.relax_y, axis=1) * self.US_SCALE
        return bounds

    def relax(self, t0=None, y0=None, w0=None):
        return self.grn(key=self.random_key, t0=t0, y0=y0, w0=w0)[0]

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
                                stimulus={stimulus: self.bounds[stimulus, int(regulation) % 2]})
        mean_x2 = np.mean(x2.ys[response, :])
        if (mean_x2 >= self.r_scale_up * np.mean(self.relax_y[response, :])
                and mean_x2 >= self.r_scale_up * np.mean(self.reference.ys[response, self.relax_t: self.relax_t * 2])):
            return MemoryCircuit(stimulus=stimulus,
                                 response=response,
                                 stimulus_reg=regulation,
                                 response_reg=Regulation(1),
                                 ys=x2.ys,
                                 ws=x2.ws,
                                 is_ucs=True)
        elif (mean_x2 <= (1 / self.r_scale_up) * np.mean(self.relax_y[response, :])
              and mean_x2 <= (1 / self.r_scale_up) * np.mean(
                    self.reference.ys[response, self.relax_t: self.relax_t * 2])):
            return MemoryCircuit(stimulus=stimulus,
                                 response=response,
                                 stimulus_reg=regulation,
                                 response_reg=Regulation(2),
                                 ys=x2.ys,
                                 ws=x2.ws,
                                 is_ucs=True)
        return MemoryCircuit(stimulus=stimulus,
                             response=response,
                             stimulus_reg=regulation,
                             response_reg=Regulation(0),
                             is_ucs=False,
                             ys=None,
                             ws=None)

    def eval_mem_for_r(self, response, exp="ass"):
        if not self.mem_circuits[response]:
            return
        cs_list = [circuit for circuit in self.mem_circuits[response] if not circuit.is_ucs]
        for ucs_circuit in [circuit for circuit in self.mem_circuits[response] if circuit.is_ucs]:
            if exp == "ass":
                self.test_associative_memory(ucs_circuit, cs_list)
            elif exp == "habit":
                self.test_habituation(ucs_circuit=ucs_circuit)

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
                                                  self.bounds[cs_circuit.stimulus, int(cs_circuit.stimulus_reg) % 2]})
            up_down_r = self.is_r_regulated(e1, cs_circuit.response)
            if int(up_down_r) != 0:
                e2 = self.relax(y0=e1.ys[:, -1],
                                w0=e1.ws[:, -1],
                                t0=self.n_secs * 2)
                e3 = self.grn.stimulate(key=self.random_key,
                                        y0=e2.ys[:, -1],
                                        w0=e2.ws[:, -1],
                                        t0=self.n_secs * 3,
                                        stimulus={ucs_circuit.stimulus:
                                                      self.bounds[
                                                          ucs_circuit.stimulus, int(ucs_circuit.stimulus_reg) % 2]})
                is_mem = self.is_memory(e3, ucs_circuit.response, up_down_r)
                if is_mem:
                    self.save_memory(e3,
                                     r=cs_circuit.response,
                                     ucs=ucs_circuit.stimulus,
                                     cs=cs_circuit.stimulus,
                                     response_reg=up_down_r,
                                     stimulus_reg=cs_circuit.stimulus_reg,
                                     exp="ass")
                del e2, e3
            del e1
        return is_mem

    def test_habituation(self, ucs_circuit, n_stim=4, scale=1.5, increment=500):
        e = None
        r = self.relax(t0=self.n_secs * 2,
                       y0=ucs_circuit.ys[:, -1],
                       w0=ucs_circuit.ws[:, -1])
        y0 = r.ys[:, -1]
        w0 = r.ws[:, -1]
        n_secs = self.n_secs * 3
        is_habit = False
        is_sens = False
        for i in range(0, n_stim * 2, 2):
            prev_e = e.copy() if e is not None else ucs_circuit
            e = self.grn.stimulate(key=self.random_key,
                                   y0=y0,
                                   w0=w0,
                                   t0=n_secs,
                                   stimulus={ucs_circuit.stimulus:
                                                 self.bounds[ucs_circuit.stimulus, int(ucs_circuit.stimulus_reg) % 2]})
            n_secs += e.ys.shape[1] * self.grn.config.deltaT
            if i == 0 or is_habit:
                is_habit = self.is_habituation(e=e,
                                               prev_e=prev_e.ys,
                                               ucs_circuit=ucs_circuit,
                                               scale=scale)
            if i == 0 or is_sens:
                is_sens = self.is_sensitization(e=e,
                                                prev_e=prev_e.ys,
                                                ucs_circuit=ucs_circuit,
                                                scale=scale)
            if not is_habit and not is_sens:
                self.grn.set_time(n_secs=self.n_secs)
                break
            r = self.relax(t0=n_secs,
                           y0=e.ys[:, -1],
                           w0=e.ws[:, -1])
            n_secs += r.ys.shape[1] * self.grn.config.deltaT
            y0 = r.ys[:, -1]
            w0 = r.ws[:, -1]
            self.grn.set_time(n_secs=self.n_secs + ((i // 2 + 1) * increment))
        else:
            if not is_habit:
                return
            self.save_memory(prev_e,
                             r=ucs_circuit.response,
                             ucs=None,
                             cs=ucs_circuit.stimulus,
                             response_reg=ucs_circuit.response_reg,
                             stimulus_reg=ucs_circuit.stimulus_reg,
                             exp="habit" if is_habit else "sens")
            self.grn.set_time(n_secs=self.n_secs)
        del e

    def is_habituation(self, e, prev_e, ucs_circuit, scale):
        mean = np.mean(e.ys[ucs_circuit.response, :])
        prev_mean = np.mean(prev_e[ucs_circuit.response, :])
        if mean < 0.0 or prev_mean < 0.0:
            return False
        elif ucs_circuit.response_reg == 1:
            return mean < prev_mean / scale
        return mean > prev_mean * scale

    def is_sensitization(self, e, prev_e, ucs_circuit, scale):
        mean = np.mean(e.ys[ucs_circuit.response, :])
        prev_mean = np.mean(prev_e[ucs_circuit.response, :])
        if mean < 0.0 or prev_mean < 0.0:
            return False
        if ucs_circuit.response_reg == 1:
            return mean > prev_mean * scale
        return mean < prev_mean / scale

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

    def save_memory(self, e, r, ucs, cs, response_reg, stimulus_reg, exp):
        # new_bounds = self.bounds.copy()
        # new_bounds[:, 0] *= self.us_scale
        # new_bounds[:, 1] /= self.us_scale
        idx = max([int(file.split(".")[2]) for file in os.listdir(os.path.join("memories", exp))
                   if file.startswith(".".join([str(self.i), str(r), ""]))] + [0]) + 1
        file_name = os.path.join("memories",
                                 exp,
                                 ".".join([str(self.i),
                                           str(r),
                                           str(idx),
                                           "pickle"]))
        pickle.dump([e.ys[:, -1],
                     e.ws[:, -1],
                     e.cs[:, -1],
                     self.bounds,
                     np.mean(self.relax_y[r, :]),
                     int(response_reg),
                     int(stimulus_reg),
                     self.r_scale_up,
                     r,
                     ucs,
                     cs],
                    open(file_name, "wb"))


def learn(seed, i, exp):
    al = AssociativeLearning(seed=seed, model_id=i)
    al.pretest()
    for r in al.mem_circuits.keys():
        al.eval_mem_for_r(response=r, exp=exp)


if __name__ == "__main__":
    # 26, 27, 29, 31
    arguments = parse_args()
    set_seed(arguments.seed)
    learn(arguments.seed, arguments.task.split("-")[0], exp=arguments.exp)
