import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([4.9, 6.33, 30.0, 0.1, 3.67, 1.0, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01, 0.2])
y_indexes = {'ADP': 0, 'NAD': 1, 'halfglucose': 2, 'ATP': 3, 'NADH': 4, 'pyruvate': 5, 'lactate': 6, 'CoA': 7, 'AcCoA': 8, 'AcP': 9, 'Ac': 10, 'AcO': 11, 'EtOH': 12, 'AcLac': 13, 'AcetoinIn': 14, 'AcetoinOut': 15, 'Butanediol': 16, 'O2': 17}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([10.0, 1.0, 2397.0, 0.1, 0.1412, 0.04699, 2.5, 0.08999, 0.01867, 5118.0, 21120.69, 1.5, 0.08, 100.0, 2.4, 259.0, 1.0, 0.4, 0.014, 46.4159, 0.1, 0.008, 42.0, 0.0065, 0.2, 2.6, 2.6, 0.2, 0.029, 0.7, 2700.0, 174.217, 0.5, 0.16, 7.0, 0.07, 97.0, 1.0, 0.007, 0.025, 0.08, 0.008, 10.0, 162.0, 12354.9, 0.03, 0.05, 0.08, 1.0, 600.0, 50.0, 9000000000000.0, 100.0, 2.4, 106.0, 10.0, 100.0, 200.0, 5.0, 105.0, 1400.0, 0.06, 0.02, 2.6, 0.16, 900.0, 6.196, 2.58, 118.0, 0.041, 0.2, 1.0, 0.0003]) 
c_indexes = {'PO4': 0, 'compartment': 1, 'R1_V_1': 2, 'R1_Kglc_1': 3, 'R1_Knad_1': 4, 'R1_Kadp_1': 5, 'R1_Kpyr_1': 6, 'R1_Knadh_1': 7, 'R1_Katp_1': 8, 'R2_V_2': 9, 'R2_Keq_2': 10, 'R2_Kpyr_2': 11, 'R2_Knadh_2': 12, 'R2_Klac_2': 13, 'R2_Knad_2': 14, 'R3_V_3': 15, 'R3_Kpyr_3': 16, 'R3_Knad_3': 17, 'R3_Kcoa_3': 18, 'R3_Ki_3': 19, 'R3_Knadh_3': 20, 'R3_Kaccoa_3': 21, 'R4_V_4': 22, 'R4_Keq_4': 23, 'R4_Kiaccoa_4': 24, 'R4_Kpi_4': 25, 'R4_Kipi_4': 26, 'R4_Kiacp_4': 27, 'R4_Kicoa_4': 28, 'R4_Kacp_4': 29, 'R5_V_5': 30, 'R5_Keq_5': 31, 'R5_Kadp_5': 32, 'R5_Kacp_5': 33, 'R5_Kac_5': 34, 'R5_Katp_5': 35, 'R6_V_6': 36, 'R6_Keq_6': 37, 'R6_Kaccoa_6': 38, 'R6_Knadh_6': 39, 'R6_Knad_6': 40, 'R6_Kcoa_6': 41, 'R6_Kaco_6': 42, 'R7_V_7': 43, 'R7_Keq_7': 44, 'R7_Kaco_7': 45, 'R7_Knadh_7': 46, 'R7_Knad_7': 47, 'R7_Ketoh_7': 48, 'R8_V_8': 49, 'R8_Kpyr_8': 50, 'R8_Keq_8': 51, 'R8_Kaclac_8': 52, 'R8_n_8': 53, 'R9_V_9': 54, 'R9_Kaclac_9': 55, 'R9_Kacet_9': 56, 'R10_V_10': 57, 'R10_Kacet_10': 58, 'R11_V_11': 59, 'R11_Keq_11': 60, 'R11_Kacet_11': 61, 'R11_Knadh_11': 62, 'R11_Kbut_11': 63, 'R11_Knad_11': 64, 'R12_V_12': 65, 'R12_Katp_12': 66, 'R12_n_12': 67, 'R13_V_13': 68, 'R13_Knadh_13': 69, 'R13_Ko_13': 70, 'R13_Knad_13': 71, 'R14_k_14': 72}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [-1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [1.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0], [1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.R1(y, w, c, t), self.R2(y, w, c, t), self.R3(y, w, c, t), self.R4(y, w, c, t), self.R5(y, w, c, t), self.R6(y, w, c, t), self.R7(y, w, c, t), self.R8(y, w, c, t), self.R9(y, w, c, t), self.R10(y, w, c, t), self.R11(y, w, c, t), self.R12(y, w, c, t), self.R13(y, w, c, t), self.R14(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def R1(self, y, w, c, t):
		return 2 * c[2] * (y[2] / (2 * c[3])) * (y[1] / c[4]) * (y[0] / c[5]) / ((1 + y[2] / (2 * c[3]) + y[5] / c[6]) * (1 + y[1] / c[4] + y[4] / c[7]) * (1 + y[0] / c[5] + y[3] / c[8]))


	def R2(self, y, w, c, t):
		return c[9] * ((y[5] * y[4] - y[6] * y[1] / c[10]) / (c[11] * c[12])) / ((1 + y[5] / c[11] + y[6] / c[13]) * (1 + y[4] / c[12] + y[1] / c[14]))


	def R3(self, y, w, c, t):
		return c[15] * (y[5] / c[16]) * (y[1] / c[17]) * (y[7] / c[18]) * (y[1] / (y[1] + c[19] * y[4])) / ((1 + y[5] / c[16]) * (1 + y[1] / c[17] + y[4] / c[20]) * (1 + y[7] / c[18] + y[8] / c[21]))


	def R4(self, y, w, c, t):
		return c[22] * ((y[8] * (c[0]/1.0) - y[9] * y[7] / c[23]) / (c[24] * c[25])) / (1 + y[8] / c[24] + (c[0]/1.0) / c[26] + y[9] / c[27] + y[7] / c[28] + y[8] * (c[0]/1.0) / (c[24] * c[25]) + y[9] * y[7] / (c[29] * c[28]))


	def R5(self, y, w, c, t):
		return c[30] * ((y[9] * y[0] - y[10] * y[3] / c[31]) / (c[32] * c[33])) / ((1 + y[9] / c[33] + y[10] / c[34]) * (1 + y[0] / c[32] + y[3] / c[35]))


	def R6(self, y, w, c, t):
		return c[36] * ((y[8] * y[4] - y[7] * y[1] * y[11] / c[37]) / (c[38] * c[39])) / ((1 + y[1] / c[40] + y[4] / c[39]) * (1 + y[8] / c[38] + y[7] / c[41]) * (1 + y[11] / c[42]))


	def R7(self, y, w, c, t):
		return c[43] * ((y[11] * y[4] - y[12] * y[1] / c[44]) / (c[45] * c[46])) / ((1 + y[1] / c[47] + y[4] / c[46]) * (1 + y[11] / c[45] + y[12] / c[48]))


	def R8(self, y, w, c, t):
		return c[49] * (y[5] / c[50]) * (1 - y[13] / (y[5] * c[51])) * (y[5] / c[50] + y[13] / c[52])**(c[53] - 1) / (1 + (y[5] / c[50] + y[13] / c[52])**c[53])


	def R9(self, y, w, c, t):
		return c[54] * (y[13] / c[55]) / (1 + y[13] / c[55] + y[14] / c[56])


	def R10(self, y, w, c, t):
		return c[57] * (y[14] / c[58]) / (1 + y[14] / c[58])


	def R11(self, y, w, c, t):
		return c[59] * ((y[14] * y[4] - y[16] * y[1] / c[60]) / (c[61] * c[62])) / ((1 + y[14] / c[61] + y[16] / c[63]) * (1 + y[4] / c[62] + y[1] / c[64]))


	def R12(self, y, w, c, t):
		return c[65] * (y[3] / (y[0] * c[66]))**c[67] / (1 + (y[3] / (y[0] * c[66]))**c[67])


	def R13(self, y, w, c, t):
		return c[68] * (y[4] * y[17] / (c[69] * c[70])) / ((1 + y[4] / c[69] + y[1] / c[71]) * (1 + y[17] / c[70]))


	def R14(self, y, w, c, t):
		return c[72] * y[13]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		return w

class ModelStep(eqx.Module):
	y_indexes: dict = eqx.static_field()
	w_indexes: dict = eqx.static_field()
	c_indexes: dict = eqx.static_field()
	ratefunc: RateofSpeciesChange
	atol: float = eqx.static_field()
	rtol: float = eqx.static_field()
	mxstep: int = eqx.static_field()
	assignmentfunc: AssignmentRule

	def __init__(self, y_indexes={'ADP': 0, 'NAD': 1, 'halfglucose': 2, 'ATP': 3, 'NADH': 4, 'pyruvate': 5, 'lactate': 6, 'CoA': 7, 'AcCoA': 8, 'AcP': 9, 'Ac': 10, 'AcO': 11, 'EtOH': 12, 'AcLac': 13, 'AcetoinIn': 14, 'AcetoinOut': 15, 'Butanediol': 16, 'O2': 17}, w_indexes={}, c_indexes={'PO4': 0, 'compartment': 1, 'R1_V_1': 2, 'R1_Kglc_1': 3, 'R1_Knad_1': 4, 'R1_Kadp_1': 5, 'R1_Kpyr_1': 6, 'R1_Knadh_1': 7, 'R1_Katp_1': 8, 'R2_V_2': 9, 'R2_Keq_2': 10, 'R2_Kpyr_2': 11, 'R2_Knadh_2': 12, 'R2_Klac_2': 13, 'R2_Knad_2': 14, 'R3_V_3': 15, 'R3_Kpyr_3': 16, 'R3_Knad_3': 17, 'R3_Kcoa_3': 18, 'R3_Ki_3': 19, 'R3_Knadh_3': 20, 'R3_Kaccoa_3': 21, 'R4_V_4': 22, 'R4_Keq_4': 23, 'R4_Kiaccoa_4': 24, 'R4_Kpi_4': 25, 'R4_Kipi_4': 26, 'R4_Kiacp_4': 27, 'R4_Kicoa_4': 28, 'R4_Kacp_4': 29, 'R5_V_5': 30, 'R5_Keq_5': 31, 'R5_Kadp_5': 32, 'R5_Kacp_5': 33, 'R5_Kac_5': 34, 'R5_Katp_5': 35, 'R6_V_6': 36, 'R6_Keq_6': 37, 'R6_Kaccoa_6': 38, 'R6_Knadh_6': 39, 'R6_Knad_6': 40, 'R6_Kcoa_6': 41, 'R6_Kaco_6': 42, 'R7_V_7': 43, 'R7_Keq_7': 44, 'R7_Kaco_7': 45, 'R7_Knadh_7': 46, 'R7_Knad_7': 47, 'R7_Ketoh_7': 48, 'R8_V_8': 49, 'R8_Kpyr_8': 50, 'R8_Keq_8': 51, 'R8_Kaclac_8': 52, 'R8_n_8': 53, 'R9_V_9': 54, 'R9_Kaclac_9': 55, 'R9_Kacet_9': 56, 'R10_V_10': 57, 'R10_Kacet_10': 58, 'R11_V_11': 59, 'R11_Keq_11': 60, 'R11_Kacet_11': 61, 'R11_Knadh_11': 62, 'R11_Kbut_11': 63, 'R11_Knad_11': 64, 'R12_V_12': 65, 'R12_Katp_12': 66, 'R12_n_12': 67, 'R13_V_13': 68, 'R13_Knadh_13': 69, 'R13_Ko_13': 70, 'R13_Knad_13': 71, 'R14_k_14': 72}, atol=1e-06, rtol=1e-12, mxstep=5000000):

		self.y_indexes = y_indexes
		self.w_indexes = w_indexes
		self.c_indexes = c_indexes

		self.ratefunc = RateofSpeciesChange()
		self.rtol = rtol
		self.atol = atol
		self.mxstep = mxstep
		self.assignmentfunc = AssignmentRule()

	@jit
	def __call__(self, y, w, c, t, deltaT):
		y_new = odeint(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]	
		t_new = t + deltaT	
		w_new = self.assignmentfunc(y_new, w, c, t_new)	
		return y_new, w_new, c, t_new	

class ModelRollout(eqx.Module):
	deltaT: float = eqx.static_field()
	modelstepfunc: ModelStep

	def __init__(self, deltaT=0.1, atol=1e-06, rtol=1e-12, mxstep=5000000):

		self.deltaT = deltaT
		self.modelstepfunc = ModelStep(atol=atol, rtol=rtol, mxstep=mxstep)

	@partial(jit, static_argnames=("n_steps",))
	def __call__(self, n_steps, y0=jnp.array([4.9, 6.33, 30.0, 0.1, 3.67, 1.0, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01, 0.2]), w0=jnp.array([]), c=jnp.array([10.0, 1.0, 2397.0, 0.1, 0.1412, 0.04699, 2.5, 0.08999, 0.01867, 5118.0, 21120.69, 1.5, 0.08, 100.0, 2.4, 259.0, 1.0, 0.4, 0.014, 46.4159, 0.1, 0.008, 42.0, 0.0065, 0.2, 2.6, 2.6, 0.2, 0.029, 0.7, 2700.0, 174.217, 0.5, 0.16, 7.0, 0.07, 97.0, 1.0, 0.007, 0.025, 0.08, 0.008, 10.0, 162.0, 12354.9, 0.03, 0.05, 0.08, 1.0, 600.0, 50.0, 9000000000000.0, 100.0, 2.4, 106.0, 10.0, 100.0, 200.0, 5.0, 105.0, 1400.0, 0.06, 0.02, 2.6, 0.16, 900.0, 6.196, 2.58, 118.0, 0.041, 0.2, 1.0, 0.0003]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

