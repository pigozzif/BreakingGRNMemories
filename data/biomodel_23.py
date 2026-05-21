import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 5.1, 0.0, 0.0])
y_indexes = {'Fruex': 0, 'Fru': 1, 'Glcex': 2, 'Glc': 3, 'ATP': 4, 'HexP': 5, 'ADP': 6, 'UDP': 7, 'Suc6P': 8, 'Suc': 9, 'phos': 10, 'glycolysis': 11, 'Sucvac': 12}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 0.286, 0.2, 1.0, 0.286, 0.2, 1.0, 0.197, 0.07, 0.25, 10.0, 0.1, 10.0, 0.197, 10.0, 0.25, 0.07, 0.1, 10.0, 0.164, 12.0, 0.1, 0.085, 2.0, 0.379, 10.0, 0.07, 0.6, 3.0, 1.4, 1.8, 0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.677, 5.0, 4.0, 50.0, 0.3, 0.3, 0.3, 0.3, 4.0, 40.0, 0.372, 15.0, 10.0, 15.0, 0.1, 0.2, 1.0, 100.0]) 
c_indexes = {'compartment': 0, 'v1_Vmax1': 1, 'v1_Km1Fruex': 2, 'v1_Ki1Fru': 3, 'v2_Vmax2': 4, 'v2_Km2Glcex': 5, 'v2_Ki2Glc': 6, 'v3_Vmax3': 7, 'v3_Km3Glc': 8, 'v3_Km3ATP': 9, 'v3_Km4Fru': 10, 'v3_Ki3G6P': 11, 'v3_Ki4F6P': 12, 'v4_Vmax4': 13, 'v4_Km4Fru': 14, 'v4_Km4ATP': 15, 'v4_Km3Glc': 16, 'v4_Ki3G6P': 17, 'v4_Ki4F6P': 18, 'v5_Vmax5': 19, 'v5_Ki5Fru': 20, 'v5_Km5Fru': 21, 'v5_Km5ATP': 22, 'v5_Ki5ADP': 23, 'v6_Vmax6f': 24, 'v6_Keq6': 25, 'v6_Ki6Suc6P': 26, 'v6_Km6F6P': 27, 'v6_Ki6Pi': 28, 'v6_Ki6UDPGlc': 29, 'v6_Km6UDPGlc': 30, 'v6_Vmax6r': 31, 'v6_Km6UDP': 32, 'v6_Km6Suc6P': 33, 'v6_Ki6F6P': 34, 'v7_Vmax7': 35, 'v7_Km7Suc6P': 36, 'v8_Vmax8f': 37, 'v8_Keq8': 38, 'v8_Ki8Fru': 39, 'v8_Km8Suc': 40, 'v8_Ki8UDP': 41, 'v8_Km8UDP': 42, 'v8_Vmax8r': 43, 'v8_Km8UDPGlc': 44, 'v8_Km8Fru': 45, 'v8_Ki8Suc': 46, 'v9_Vmax9': 47, 'v9_Ki9Glc': 48, 'v9_Km9Suc': 49, 'v9_Ki9Fru': 50, 'v10_Vmax10': 51, 'v10_Km10F6P': 52, 'v11_Vmax11': 53, 'v11_Km11Suc': 54}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0, -2.0, 0.0, -1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.v5(y, w, c, t), self.v6(y, w, c, t), self.v7(y, w, c, t), self.v8(y, w, c, t), self.v9(y, w, c, t), self.v10(y, w, c, t), self.v11(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[0] * (c[1] * (y[0]/1.0) / (c[2] * (1 + (y[1]/1.0) / c[3]) + (y[0]/1.0)))


	def v2(self, y, w, c, t):
		return c[0] * (c[4] * (y[2]/1.0) / (c[5] * (1 + (y[3]/1.0) / c[6]) + (y[2]/1.0)))


	def v3(self, y, w, c, t):
		return c[0] * (c[7] * ((y[3]/1.0) / c[8]) * ((y[4]/1.0) / c[9]) / ((1 + (y[4]/1.0) / c[9]) * (1 + (y[3]/1.0) / c[8] + (y[1]/1.0) / c[10] + 0.113 * (y[5]/1.0) / c[11] + 0.0575 * (y[5]/1.0) / c[12])))


	def v4(self, y, w, c, t):
		return c[0] * (c[13] * ((y[1]/1.0) / c[14]) * ((y[4]/1.0) / c[15]) / ((1 + (y[4]/1.0) / c[15]) * (1 + (y[3]/1.0) / c[16] + (y[1]/1.0) / c[14] + 0.113 * (y[5]/1.0) / c[17] + 0.0575 * (y[5]/1.0) / c[18])))


	def v5(self, y, w, c, t):
		return c[0] * ((c[19] / (1 + (y[1]/1.0) / c[20])) * ((y[1]/1.0) / c[21]) * ((y[4]/1.0) / c[22]) / (1 + (y[1]/1.0) / c[21] + (y[4]/1.0) / c[22] + (y[1]/1.0) * (y[4]/1.0) / (c[21] * c[22]) + (y[6]/1.0) / c[23]))


	def v6(self, y, w, c, t):
		return c[0] * (c[24] * (0.0575 * (y[5]/1.0) * 0.8231 * (y[5]/1.0) - (y[8]/1.0) * (y[7]/1.0) / c[25]) / (0.0575 * (y[5]/1.0) * 0.8231 * (y[5]/1.0) * (1 + (y[8]/1.0) / c[26]) + c[27] * (1 + (y[10]/1.0) / c[28]) * (0.8231 * (y[5]/1.0) + c[29]) + c[30] * 0.0575 * (y[5]/1.0) + (c[24] / (c[31] * c[25])) * (c[32] * (y[8]/1.0) * (1 + 0.8231 * (y[5]/1.0) / c[29]) + (y[7]/1.0) * (c[33] * (1 + c[30] * 0.0575 * (y[5]/1.0) / (c[29] * c[27] * (1 + (y[10]/1.0) / c[28]))) + (y[8]/1.0) * (1 + 0.0575 * (y[5]/1.0) / c[34])))))


	def v7(self, y, w, c, t):
		return c[0] * (c[35] * (y[8]/1.0) / (c[36] + (y[8]/1.0)))


	def v8(self, y, w, c, t):
		return c[0] * (-c[37] * ((y[9]/1.0) * (y[7]/1.0) - (y[1]/1.0) * 0.8231 * (y[5]/1.0) / c[38]) / ((y[9]/1.0) * (y[7]/1.0) * (1 + (y[1]/1.0) / c[39]) + c[40] * ((y[7]/1.0) + c[41]) + c[42] * (y[9]/1.0) + (c[37] / (c[43] * c[38])) * (c[44] * (y[1]/1.0) * (1 + (y[7]/1.0) / c[41]) + 0.8231 * (y[5]/1.0) * (c[45] * (1 + c[42] * (y[9]/1.0) / (c[41] * c[40])) + (y[1]/1.0) * (1 + (y[9]/1.0) / c[46])))))


	def v9(self, y, w, c, t):
		return c[0] * ((c[47] / (1 + (y[3]/1.0) / c[48])) * (y[9]/1.0) / (c[49] * (1 + (y[1]/1.0) / c[50]) + (y[9]/1.0)))


	def v10(self, y, w, c, t):
		return c[0] * (c[51] * 0.0575 * (y[5]/1.0) / (c[52] + 0.0575 * (y[5]/1.0)))


	def v11(self, y, w, c, t):
		return c[0] * (c[53] * (y[9]/1.0) / (c[54] + (y[9]/1.0)))

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

	def __init__(self, y_indexes={'Fruex': 0, 'Fru': 1, 'Glcex': 2, 'Glc': 3, 'ATP': 4, 'HexP': 5, 'ADP': 6, 'UDP': 7, 'Suc6P': 8, 'Suc': 9, 'phos': 10, 'glycolysis': 11, 'Sucvac': 12}, w_indexes={}, c_indexes={'compartment': 0, 'v1_Vmax1': 1, 'v1_Km1Fruex': 2, 'v1_Ki1Fru': 3, 'v2_Vmax2': 4, 'v2_Km2Glcex': 5, 'v2_Ki2Glc': 6, 'v3_Vmax3': 7, 'v3_Km3Glc': 8, 'v3_Km3ATP': 9, 'v3_Km4Fru': 10, 'v3_Ki3G6P': 11, 'v3_Ki4F6P': 12, 'v4_Vmax4': 13, 'v4_Km4Fru': 14, 'v4_Km4ATP': 15, 'v4_Km3Glc': 16, 'v4_Ki3G6P': 17, 'v4_Ki4F6P': 18, 'v5_Vmax5': 19, 'v5_Ki5Fru': 20, 'v5_Km5Fru': 21, 'v5_Km5ATP': 22, 'v5_Ki5ADP': 23, 'v6_Vmax6f': 24, 'v6_Keq6': 25, 'v6_Ki6Suc6P': 26, 'v6_Km6F6P': 27, 'v6_Ki6Pi': 28, 'v6_Ki6UDPGlc': 29, 'v6_Km6UDPGlc': 30, 'v6_Vmax6r': 31, 'v6_Km6UDP': 32, 'v6_Km6Suc6P': 33, 'v6_Ki6F6P': 34, 'v7_Vmax7': 35, 'v7_Km7Suc6P': 36, 'v8_Vmax8f': 37, 'v8_Keq8': 38, 'v8_Ki8Fru': 39, 'v8_Km8Suc': 40, 'v8_Ki8UDP': 41, 'v8_Km8UDP': 42, 'v8_Vmax8r': 43, 'v8_Km8UDPGlc': 44, 'v8_Km8Fru': 45, 'v8_Ki8Suc': 46, 'v9_Vmax9': 47, 'v9_Ki9Glc': 48, 'v9_Km9Suc': 49, 'v9_Ki9Fru': 50, 'v10_Vmax10': 51, 'v10_Km10F6P': 52, 'v11_Vmax11': 53, 'v11_Km11Suc': 54}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0, 5.1, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 0.286, 0.2, 1.0, 0.286, 0.2, 1.0, 0.197, 0.07, 0.25, 10.0, 0.1, 10.0, 0.197, 10.0, 0.25, 0.07, 0.1, 10.0, 0.164, 12.0, 0.1, 0.085, 2.0, 0.379, 10.0, 0.07, 0.6, 3.0, 1.4, 1.8, 0.2, 0.3, 0.1, 0.4, 0.5, 0.1, 0.677, 5.0, 4.0, 50.0, 0.3, 0.3, 0.3, 0.3, 4.0, 40.0, 0.372, 15.0, 10.0, 15.0, 0.1, 0.2, 1.0, 100.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

