import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1e-22, 1e-21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'B': 0, 'L': 1, 'BL': 2, 'BLL': 3, 'ALL': 4, 'A': 5, 'AL': 6, 'I': 7, 'IL': 8, 'ILL': 9, 'D': 10, 'DL': 11, 'DLL': 12}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([300000000.0, 8000.0, 150000000.0, 16000.0, 30000.0, 700.0, 300000000.0, 8.64, 150000000.0, 17.28, 0.54, 10800.0, 130.0, 2740.0, 300000000.0, 4.0, 150000000.0, 8.0, 19.7, 3.74, 19.85, 1.74, 20.0, 0.81, 300000000.0, 4.0, 150000000.0, 8.0, 0.05, 0.0012, 0.05, 0.0012, 0.05, 0.0012, 1e-16]) 
c_indexes = {'kf_0': 0, 'kr_0': 1, 'kf_1': 2, 'kr_1': 3, 'kf_2': 4, 'kr_2': 5, 'kf_3': 6, 'kr_3': 7, 'kf_4': 8, 'kr_4': 9, 'kf_5': 10, 'kr_5': 11, 'kf_6': 12, 'kr_6': 13, 'kf_7': 14, 'kr_7': 15, 'kf_8': 16, 'kr_8': 17, 'kf_9': 18, 'kr_9': 19, 'kf_10': 20, 'kr_10': 21, 'kf_11': 22, 'kr_11': 23, 'kf_12': 24, 'kr_12': 25, 'kf_13': 26, 'kr_13': 27, 'kf_14': 28, 'kr_14': 29, 'kf_15': 30, 'kr_15': 31, 'kf_16': 32, 'kr_16': 33, 'comp1': 34}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.React0(y, w, c, t), self.React1(y, w, c, t), self.React2(y, w, c, t), self.React3(y, w, c, t), self.React4(y, w, c, t), self.React5(y, w, c, t), self.React6(y, w, c, t), self.React7(y, w, c, t), self.React8(y, w, c, t), self.React9(y, w, c, t), self.React10(y, w, c, t), self.React11(y, w, c, t), self.React12(y, w, c, t), self.React13(y, w, c, t), self.React14(y, w, c, t), self.React15(y, w, c, t), self.React16(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def React0(self, y, w, c, t):
		return c[34] * (c[0] * (y[0]/1e-16) * (y[1]/1e-16) - c[1] * (y[2]/1e-16))


	def React1(self, y, w, c, t):
		return c[34] * (c[2] * (y[2]/1e-16) * (y[1]/1e-16) - c[3] * (y[3]/1e-16))


	def React2(self, y, w, c, t):
		return c[34] * (c[4] * (y[3]/1e-16) - c[5] * (y[4]/1e-16))


	def React3(self, y, w, c, t):
		return c[34] * (c[6] * (y[5]/1e-16) * (y[1]/1e-16) - c[7] * (y[6]/1e-16))


	def React4(self, y, w, c, t):
		return c[34] * (c[8] * (y[6]/1e-16) * (y[1]/1e-16) - c[9] * (y[4]/1e-16))


	def React5(self, y, w, c, t):
		return c[34] * (c[10] * (y[0]/1e-16) - c[11] * (y[5]/1e-16))


	def React6(self, y, w, c, t):
		return c[34] * (c[12] * (y[2]/1e-16) - c[13] * (y[6]/1e-16))


	def React7(self, y, w, c, t):
		return c[34] * (c[14] * (y[7]/1e-16) * (y[1]/1e-16) - c[15] * (y[8]/1e-16))


	def React8(self, y, w, c, t):
		return c[34] * (c[16] * (y[8]/1e-16) * (y[1]/1e-16) - c[17] * (y[9]/1e-16))


	def React9(self, y, w, c, t):
		return c[34] * (c[18] * (y[5]/1e-16) - c[19] * (y[7]/1e-16))


	def React10(self, y, w, c, t):
		return c[34] * (c[20] * (y[6]/1e-16) - c[21] * (y[8]/1e-16))


	def React11(self, y, w, c, t):
		return c[34] * (c[22] * (y[4]/1e-16) - c[23] * (y[9]/1e-16))


	def React12(self, y, w, c, t):
		return c[34] * (c[24] * (y[10]/1e-16) * (y[1]/1e-16) - c[25] * (y[11]/1e-16))


	def React13(self, y, w, c, t):
		return c[34] * (c[26] * (y[11]/1e-16) * (y[1]/1e-16) - c[27] * (y[12]/1e-16))


	def React14(self, y, w, c, t):
		return c[34] * (c[28] * (y[7]/1e-16) - c[29] * (y[10]/1e-16))


	def React15(self, y, w, c, t):
		return c[34] * (c[30] * (y[8]/1e-16) - c[31] * (y[11]/1e-16))


	def React16(self, y, w, c, t):
		return c[34] * (c[32] * (y[9]/1e-16) - c[33] * (y[12]/1e-16))

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

	def __init__(self, y_indexes={'B': 0, 'L': 1, 'BL': 2, 'BLL': 3, 'ALL': 4, 'A': 5, 'AL': 6, 'I': 7, 'IL': 8, 'ILL': 9, 'D': 10, 'DL': 11, 'DLL': 12}, w_indexes={}, c_indexes={'kf_0': 0, 'kr_0': 1, 'kf_1': 2, 'kr_1': 3, 'kf_2': 4, 'kr_2': 5, 'kf_3': 6, 'kr_3': 7, 'kf_4': 8, 'kr_4': 9, 'kf_5': 10, 'kr_5': 11, 'kf_6': 12, 'kr_6': 13, 'kf_7': 14, 'kr_7': 15, 'kf_8': 16, 'kr_8': 17, 'kf_9': 18, 'kr_9': 19, 'kf_10': 20, 'kr_10': 21, 'kf_11': 22, 'kr_11': 23, 'kf_12': 24, 'kr_12': 25, 'kf_13': 26, 'kr_13': 27, 'kf_14': 28, 'kr_14': 29, 'kf_15': 30, 'kr_15': 31, 'kf_16': 32, 'kr_16': 33, 'comp1': 34}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([1e-22, 1e-21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([300000000.0, 8000.0, 150000000.0, 16000.0, 30000.0, 700.0, 300000000.0, 8.64, 150000000.0, 17.28, 0.54, 10800.0, 130.0, 2740.0, 300000000.0, 4.0, 150000000.0, 8.0, 19.7, 3.74, 19.85, 1.74, 20.0, 0.81, 300000000.0, 4.0, 150000000.0, 8.0, 0.05, 0.0012, 0.05, 0.0012, 0.05, 0.0012, 1e-16]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

