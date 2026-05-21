import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
y_indexes = {'CL_m': 0, 'CL_p': 1, 'P97_m': 2, 'P97_p': 3, 'P51_m': 4, 'P51_p': 5, 'EL_m': 6, 'EL_p': 7, 'P': 8, 'PIF_m': 9, 'PIF_p': 10, 'hypocotyl': 11}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([4.58, 3.0, 1.27, 1.48, 1.0, 1.47, 0.53, 5.0, 0.21, 0.35, 0.56, 0.57, 0.76, 0.42, 1.01, 0.64, 1.01, 0.68, 0.5, 0.29, 0.48, 0.78, 1.21, 0.38, 0.16, 1.18, 0.24, 0.23, 0.3, 0.46, 2.0, 0.36, 1.9, 1.9, 0.1, 0.14, 0.62, 4.0, 0.52, 0.01, 0.12, 0.21, 0.56, 12.0, 24.0, 1.0, 0.0, 1.0]) 
c_indexes = {'v1': 0, 'v1L': 1, 'v2A': 2, 'v2B': 3, 'v3': 4, 'v4': 5, 'k1L': 6, 'v2L': 7, 'k1D': 8, 'k2': 9, 'k3': 10, 'k4': 11, 'p1': 12, 'p1L': 13, 'p2': 14, 'p3': 15, 'p4': 16, 'd1': 17, 'd2D': 18, 'd2L': 19, 'd3D': 20, 'd3L': 21, 'd4D': 22, 'd4L': 23, 'K1': 24, 'K2': 25, 'K3': 26, 'K4': 27, 'K5': 28, 'K6': 29, 'K7': 30, 'K8': 31, 'K9': 32, 'K10': 33, 'v5': 34, 'k5': 35, 'p5': 36, 'd5L': 37, 'd5D': 38, 'g1': 39, 'g2': 40, 'K11': 41, 'K12': 42, 'PP': 43, 'T': 44, 'L': 45, 'D': 46, 'cell': 47}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.CL_transcription(y, w, c, t), self.CL_light_transcription(y, w, c, t), self.CLm_light_degradation(y, w, c, t), self.CLm_dark_degradation(y, w, c, t), self.CL_translation(y, w, c, t), self.CL_light_translation(y, w, c, t), self.CLp_degradation(y, w, c, t), self.P97_light_transcription(y, w, c, t), self.P97_transcription(y, w, c, t), self.P97_CL_transcription(y, w, c, t), self.P97m_degradation(y, w, c, t), self.P97_translation(y, w, c, t), self.P97_dark_degradation(y, w, c, t), self.P97_light_degradation(y, w, c, t), self.P51_transcription(y, w, c, t), self.P51m_degradation(y, w, c, t), self.P51_translation(y, w, c, t), self.P51_dark_degradation(y, w, c, t), self.P51_light_degradation(y, w, c, t), self.EL_light_transcription(y, w, c, t), self.ELm_degradation(y, w, c, t), self.EL_translation(y, w, c, t), self.EL_dark_degradation(y, w, c, t), self.EL_light_degradation(y, w, c, t), self.P_dark_accumulation(y, w, c, t), self.P_light_degradation(y, w, c, t), self.PIF_transcription(y, w, c, t), self.PIFm_degradation(y, w, c, t), self.PIF_translation(y, w, c, t), self.PIF_dark_degradation(y, w, c, t), self.PIF_light_degradation(y, w, c, t), self.basal_growth(y, w, c, t), self.PIF_induced_growth(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def CL_transcription(self, y, w, c, t):
		return c[0] / (1 + ((y[3]/1.0) / c[24])**2 + ((y[5]/1.0) / c[25])**2)


	def CL_light_transcription(self, y, w, c, t):
		return c[1] * c[45] * (y[8]/1.0) / (1 + ((y[3]/1.0) / c[24])**2 + ((y[5]/1.0) / c[25])**2)


	def CLm_light_degradation(self, y, w, c, t):
		return c[6] * c[45] * (y[0]/1.0)


	def CLm_dark_degradation(self, y, w, c, t):
		return c[8] * c[46] * (y[0]/1.0)


	def CL_translation(self, y, w, c, t):
		return c[12] * (y[0]/1.0)


	def CL_light_translation(self, y, w, c, t):
		return c[13] * c[45] * (y[0]/1.0)


	def CLp_degradation(self, y, w, c, t):
		return c[17] * (y[1]/1.0)


	def P97_light_transcription(self, y, w, c, t):
		return c[7] * c[45] * (y[8]/1.0) / (1 + ((y[5]/1.0) / c[27])**2 + ((y[7]/1.0) / c[28])**2)


	def P97_transcription(self, y, w, c, t):
		return c[2] / (1 + ((y[5]/1.0) / c[27])**2 + ((y[7]/1.0) / c[28])**2)


	def P97_CL_transcription(self, y, w, c, t):
		return c[3] * (y[1]/1.0)**2 / (c[26]**2 + (y[1]/1.0)**2) / (1 + ((y[5]/1.0) / c[27])**2 + ((y[7]/1.0) / c[28])**2)


	def P97m_degradation(self, y, w, c, t):
		return c[9] * (y[2]/1.0)


	def P97_translation(self, y, w, c, t):
		return c[14] * (y[2]/1.0)


	def P97_dark_degradation(self, y, w, c, t):
		return c[18] * c[46] * (y[3]/1.0)


	def P97_light_degradation(self, y, w, c, t):
		return c[19] * c[45] * (y[3]/1.0)


	def P51_transcription(self, y, w, c, t):
		return c[4] / (1 + ((y[1]/1.0) / c[29])**2 + ((y[5]/1.0) / c[30])**2)


	def P51m_degradation(self, y, w, c, t):
		return c[10] * (y[4]/1.0)


	def P51_translation(self, y, w, c, t):
		return c[15] * (y[4]/1.0)


	def P51_dark_degradation(self, y, w, c, t):
		return c[20] * c[46] * (y[5]/1.0)


	def P51_light_degradation(self, y, w, c, t):
		return c[21] * c[45] * (y[5]/1.0)


	def EL_light_transcription(self, y, w, c, t):
		return c[45] * c[5] / (1 + ((y[1]/1.0) / c[31])**2 + ((y[5]/1.0) / c[32])**2 + ((y[7]/1.0) / c[33])**2)


	def ELm_degradation(self, y, w, c, t):
		return c[11] * (y[6]/1.0)


	def EL_translation(self, y, w, c, t):
		return c[16] * (y[6]/1.0)


	def EL_dark_degradation(self, y, w, c, t):
		return c[22] * c[46] * (y[7]/1.0)


	def EL_light_degradation(self, y, w, c, t):
		return c[23] * c[45] * (y[7]/1.0)


	def P_dark_accumulation(self, y, w, c, t):
		return 0.3 * (1 - (y[8]/1.0)) * c[46]


	def P_light_degradation(self, y, w, c, t):
		return (y[8]/1.0) * c[45]


	def PIF_transcription(self, y, w, c, t):
		return c[34] / (1 + ((y[7]/1.0) / c[41])**2)


	def PIFm_degradation(self, y, w, c, t):
		return c[35] * (y[9]/1.0)


	def PIF_translation(self, y, w, c, t):
		return c[36] * (y[9]/1.0)


	def PIF_dark_degradation(self, y, w, c, t):
		return c[38] * c[46] * (y[10]/1.0)


	def PIF_light_degradation(self, y, w, c, t):
		return c[37] * c[45] * (y[10]/1.0)


	def basal_growth(self, y, w, c, t):
		return c[39]


	def PIF_induced_growth(self, y, w, c, t):
		return c[40] * (y[10]/1.0)**2 / (c[42]**2 + (y[10]/1.0)**2)

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

	def __init__(self, y_indexes={'CL_m': 0, 'CL_p': 1, 'P97_m': 2, 'P97_p': 3, 'P51_m': 4, 'P51_p': 5, 'EL_m': 6, 'EL_p': 7, 'P': 8, 'PIF_m': 9, 'PIF_p': 10, 'hypocotyl': 11}, w_indexes={}, c_indexes={'v1': 0, 'v1L': 1, 'v2A': 2, 'v2B': 3, 'v3': 4, 'v4': 5, 'k1L': 6, 'v2L': 7, 'k1D': 8, 'k2': 9, 'k3': 10, 'k4': 11, 'p1': 12, 'p1L': 13, 'p2': 14, 'p3': 15, 'p4': 16, 'd1': 17, 'd2D': 18, 'd2L': 19, 'd3D': 20, 'd3L': 21, 'd4D': 22, 'd4L': 23, 'K1': 24, 'K2': 25, 'K3': 26, 'K4': 27, 'K5': 28, 'K6': 29, 'K7': 30, 'K8': 31, 'K9': 32, 'K10': 33, 'v5': 34, 'k5': 35, 'p5': 36, 'd5L': 37, 'd5D': 38, 'g1': 39, 'g2': 40, 'K11': 41, 'K12': 42, 'PP': 43, 'T': 44, 'L': 45, 'D': 46, 'cell': 47}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w0=jnp.array([]), c=jnp.array([4.58, 3.0, 1.27, 1.48, 1.0, 1.47, 0.53, 5.0, 0.21, 0.35, 0.56, 0.57, 0.76, 0.42, 1.01, 0.64, 1.01, 0.68, 0.5, 0.29, 0.48, 0.78, 1.21, 0.38, 0.16, 1.18, 0.24, 0.23, 0.3, 0.46, 2.0, 0.36, 1.9, 1.9, 0.1, 0.14, 0.62, 4.0, 0.52, 0.01, 0.12, 0.21, 0.56, 12.0, 24.0, 1.0, 0.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

