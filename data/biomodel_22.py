import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.5, 0.7, 0.1, 0.3, 0.4, 1.0, 0.9, 0.2, 0.6, 0.8])
y_indexes = {'EmptySet': 0, 'Perm': 1, 'Timm': 2, 'Clkm': 3, 'CCc': 4, 'CCn': 5, 'PTn': 6, 'PTc': 7, 'Clkc': 8, 'Perc': 9, 'Timc': 10}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.45, 0.0, 0.0, 1.02, 1.45, 4.0, 0.012, 1.0, 0.45, 0.0, 0.0, 1.02, 1.45, 4.0, 0.012, 1.0, 0.8, 0.6, 0.0, 0.89, 1.63, 4.0, 0.012, 2.0, 1.63, 2.0, 0.52, 2.0, 0.72, 2.0, 1.73, 1.63, 1.63, 1.45, 1.45, 0.48, 0.47, 0.48, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.94, 0.3, 0.44, 0.2, 0.94, 0.3, 0.44, 0.2, 0.44, 0.2, 0.29, 0.2, 0.54, 0.13, 0.6, 0.2, 0.6, 0.2, 0.3, 0.2]) 
c_indexes = {'species_0000012': 0, 'species_0000013': 1, 'Drosophilia': 2, 'compartment_0000003': 3, 'compartment_0000002': 4, 'Reaction1_a': 5, 'Reaction1_A1': 6, 'Reaction1_B1': 7, 'Reaction1_c1': 8, 'Reaction1_r1': 9, 'Reaction1_s1': 10, 'Reaction1_r': 11, 'Reaction2_D0': 12, 'Reaction3_a': 13, 'Reaction3_A2': 14, 'Reaction3_B2': 15, 'Reaction3_c2': 16, 'Reaction3_r2': 17, 'Reaction3_s3': 18, 'Reaction3_r': 19, 'Reaction4_D0': 20, 'Reaction5_a': 21, 'Reaction5_A3': 22, 'Reaction5_B3': 23, 'Reaction5_c3': 24, 'Reaction5_r3': 25, 'Reaction5_s5': 26, 'Reaction5_r': 27, 'Reaction6_D0': 28, 'Reaction7_k3': 29, 'Reaction7_T3': 30, 'Reaction8_k4': 31, 'Reaction8_T4': 32, 'Reaction9_k2': 33, 'Reaction9_T2': 34, 'Reaction10_k1': 35, 'Reaction10_T1': 36, 'Reaction11_v3': 37, 'Reaction11_parameter_0000073': 38, 'Reaction12_v1': 39, 'Reaction12_parameter_0000072': 40, 'Reaction16_s4': 41, 'Reaction18_s6': 42, 'Reaction19_s2': 43, 'Reaction20_D0': 44, 'Reaction21_D0': 45, 'Reaction23_D0': 46, 'Reaction24_D0': 47, 'Reaction25_D0': 48, 'Reaction26_D0': 49, 'Reaction27_D0': 50, 'Reaction28_D1': 51, 'Reaction28_L1': 52, 'Reaction29_D2': 53, 'Reaction29_L2': 54, 'Reaction30_D3': 55, 'Reaction30_L3': 56, 'Reaction31_D4': 57, 'Reaction31_L4': 58, 'Reaction32_D5': 59, 'Reaction32_L5': 60, 'Reaction33_D6': 61, 'Reaction33_L6': 62, 'Reaction34_D7': 63, 'Reaction34_L7': 64, 'Reaction35_D8': 65, 'Reaction35_L8': 66, 'Reaction36_D9': 67, 'Reaction36_L9': 68, 'Reaction37_D10': 69, 'Reaction37_L10': 70}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.Reaction1(y, w, c, t), self.Reaction2(y, w, c, t), self.Reaction3(y, w, c, t), self.Reaction4(y, w, c, t), self.Reaction5(y, w, c, t), self.Reaction6(y, w, c, t), self.Reaction7(y, w, c, t), self.Reaction8(y, w, c, t), self.Reaction9(y, w, c, t), self.Reaction10(y, w, c, t), self.Reaction11(y, w, c, t), self.Reaction12(y, w, c, t), self.Reaction16(y, w, c, t), self.Reaction18(y, w, c, t), self.Reaction19(y, w, c, t), self.Reaction20(y, w, c, t), self.Reaction21(y, w, c, t), self.Reaction23(y, w, c, t), self.Reaction24(y, w, c, t), self.Reaction25(y, w, c, t), self.Reaction26(y, w, c, t), self.Reaction27(y, w, c, t), self.Reaction28(y, w, c, t), self.Reaction29(y, w, c, t), self.Reaction30(y, w, c, t), self.Reaction31(y, w, c, t), self.Reaction32(y, w, c, t), self.Reaction33(y, w, c, t), self.Reaction34(y, w, c, t), self.Reaction35(y, w, c, t), self.Reaction36(y, w, c, t), self.Reaction37(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def Reaction1(self, y, w, c, t):
		return c[3] * (c[8] + (c[7] + ((y[5]/1.0) / c[6])**c[5]) * c[10] / (1 + c[7] + ((y[5]/1.0) / c[6])**c[5] + ((y[6]/1.0) / c[9])**c[11]))


	def Reaction2(self, y, w, c, t):
		return c[3] * c[12] * (y[1]/1.0)


	def Reaction3(self, y, w, c, t):
		return c[3] * (c[16] + (c[15] + ((y[5]/1.0) / c[14])**c[13]) * c[18] / (1 + c[15] + ((y[5]/1.0) / c[14])**c[13] + ((y[6]/1.0) / c[17])**c[19]))


	def Reaction4(self, y, w, c, t):
		return c[2] * c[20] * (y[2]/1.0)


	def Reaction5(self, y, w, c, t):
		return c[3] * (c[24] + (c[23] + ((y[6]/1.0) / c[22])**c[21]) * c[26] / (1 + c[23] + ((y[6]/1.0) / c[22])**c[21] + ((y[5]/1.0) / c[25])**c[27]))


	def Reaction6(self, y, w, c, t):
		return c[2] * (y[3]/1.0) * c[28]


	def Reaction7(self, y, w, c, t):
		return c[3] * ((y[4]/1.0) * c[30] / (c[29] + (y[4]/1.0)))


	def Reaction8(self, y, w, c, t):
		return c[4] * ((y[5]/1.0) * c[32] / (c[31] + (y[5]/1.0)))


	def Reaction9(self, y, w, c, t):
		return c[4] * ((y[6]/1.0) * c[34] / (c[33] + (y[6]/1.0)))


	def Reaction10(self, y, w, c, t):
		return c[3] * ((y[7]/1.0) * c[36] / (c[35] + (y[7]/1.0)))


	def Reaction11(self, y, w, c, t):
		return c[3] * ((y[8]/1.0) * c[37] * (c[0]/1.0) - c[38] * (y[4]/1.0))


	def Reaction12(self, y, w, c, t):
		return c[3] * ((y[9]/1.0) * (y[10]/1.0) * c[39] - c[40] * (y[7]/1.0))


	def Reaction16(self, y, w, c, t):
		return c[3] * c[41] * (y[2]/1.0)


	def Reaction18(self, y, w, c, t):
		return c[3] * (y[3]/1.0) * c[42]


	def Reaction19(self, y, w, c, t):
		return c[3] * c[43] * (y[1]/1.0)


	def Reaction20(self, y, w, c, t):
		return c[2] * c[44] * (y[9]/1.0)


	def Reaction21(self, y, w, c, t):
		return c[3] * c[45] * (y[7]/1.0)


	def Reaction23(self, y, w, c, t):
		return c[4] * c[46] * (y[6]/1.0)


	def Reaction24(self, y, w, c, t):
		return c[3] * (y[4]/1.0) * c[47]


	def Reaction25(self, y, w, c, t):
		return c[3] * (y[8]/1.0) * c[48]


	def Reaction26(self, y, w, c, t):
		return c[4] * (y[5]/1.0) * c[49]


	def Reaction27(self, y, w, c, t):
		return c[3] * c[50] * (y[10]/1.0)


	def Reaction28(self, y, w, c, t):
		return c[3] * (c[51] * (y[1]/1.0) / (c[52] + (y[1]/1.0)))


	def Reaction29(self, y, w, c, t):
		return c[3] * (c[53] * (c[1]/1.0) * (y[9]/1.0) / (c[54] + (y[9]/1.0)))


	def Reaction30(self, y, w, c, t):
		return c[3] * (c[55] * (y[2]/1.0) / (c[56] + (y[2]/1.0)))


	def Reaction31(self, y, w, c, t):
		return c[3] * (c[57] * (y[10]/1.0) / (c[58] + (y[10]/1.0)))


	def Reaction32(self, y, w, c, t):
		return c[3] * (c[59] * (y[7]/1.0) / (c[60] + (y[7]/1.0)))


	def Reaction33(self, y, w, c, t):
		return c[4] * (c[61] * (y[6]/1.0) / (c[62] + (y[6]/1.0)))


	def Reaction34(self, y, w, c, t):
		return c[3] * ((y[3]/1.0) * c[63] / ((y[3]/1.0) + c[64]))


	def Reaction35(self, y, w, c, t):
		return c[3] * ((y[8]/1.0) * c[65] / ((y[8]/1.0) + c[66]))


	def Reaction36(self, y, w, c, t):
		return c[3] * ((y[4]/1.0) * c[67] / ((y[4]/1.0) + c[68]))


	def Reaction37(self, y, w, c, t):
		return c[4] * ((y[5]/1.0) * c[69] / ((y[5]/1.0) + c[70]))

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

	def __init__(self, y_indexes={'EmptySet': 0, 'Perm': 1, 'Timm': 2, 'Clkm': 3, 'CCc': 4, 'CCn': 5, 'PTn': 6, 'PTc': 7, 'Clkc': 8, 'Perc': 9, 'Timc': 10}, w_indexes={}, c_indexes={'species_0000012': 0, 'species_0000013': 1, 'Drosophilia': 2, 'compartment_0000003': 3, 'compartment_0000002': 4, 'Reaction1_a': 5, 'Reaction1_A1': 6, 'Reaction1_B1': 7, 'Reaction1_c1': 8, 'Reaction1_r1': 9, 'Reaction1_s1': 10, 'Reaction1_r': 11, 'Reaction2_D0': 12, 'Reaction3_a': 13, 'Reaction3_A2': 14, 'Reaction3_B2': 15, 'Reaction3_c2': 16, 'Reaction3_r2': 17, 'Reaction3_s3': 18, 'Reaction3_r': 19, 'Reaction4_D0': 20, 'Reaction5_a': 21, 'Reaction5_A3': 22, 'Reaction5_B3': 23, 'Reaction5_c3': 24, 'Reaction5_r3': 25, 'Reaction5_s5': 26, 'Reaction5_r': 27, 'Reaction6_D0': 28, 'Reaction7_k3': 29, 'Reaction7_T3': 30, 'Reaction8_k4': 31, 'Reaction8_T4': 32, 'Reaction9_k2': 33, 'Reaction9_T2': 34, 'Reaction10_k1': 35, 'Reaction10_T1': 36, 'Reaction11_v3': 37, 'Reaction11_parameter_0000073': 38, 'Reaction12_v1': 39, 'Reaction12_parameter_0000072': 40, 'Reaction16_s4': 41, 'Reaction18_s6': 42, 'Reaction19_s2': 43, 'Reaction20_D0': 44, 'Reaction21_D0': 45, 'Reaction23_D0': 46, 'Reaction24_D0': 47, 'Reaction25_D0': 48, 'Reaction26_D0': 49, 'Reaction27_D0': 50, 'Reaction28_D1': 51, 'Reaction28_L1': 52, 'Reaction29_D2': 53, 'Reaction29_L2': 54, 'Reaction30_D3': 55, 'Reaction30_L3': 56, 'Reaction31_D4': 57, 'Reaction31_L4': 58, 'Reaction32_D5': 59, 'Reaction32_L5': 60, 'Reaction33_D6': 61, 'Reaction33_L6': 62, 'Reaction34_D7': 63, 'Reaction34_L7': 64, 'Reaction35_D8': 65, 'Reaction35_L8': 66, 'Reaction36_D9': 67, 'Reaction36_L9': 68, 'Reaction37_D10': 69, 'Reaction37_L10': 70}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.5, 0.7, 0.1, 0.3, 0.4, 1.0, 0.9, 0.2, 0.6, 0.8]), w0=jnp.array([]), c=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.45, 0.0, 0.0, 1.02, 1.45, 4.0, 0.012, 1.0, 0.45, 0.0, 0.0, 1.02, 1.45, 4.0, 0.012, 1.0, 0.8, 0.6, 0.0, 0.89, 1.63, 4.0, 0.012, 2.0, 1.63, 2.0, 0.52, 2.0, 0.72, 2.0, 1.73, 1.63, 1.63, 1.45, 1.45, 0.48, 0.47, 0.48, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.94, 0.3, 0.44, 0.2, 0.94, 0.3, 0.44, 0.2, 0.44, 0.2, 0.29, 0.2, 0.54, 0.13, 0.6, 0.2, 0.6, 0.2, 0.3, 0.2]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

