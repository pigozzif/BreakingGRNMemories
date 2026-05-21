import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.1, 0.1, 0.1, 1.0])
y_indexes = {'RA': 0, 'M_C': 1, 'C': 2, 'F': 3}

w0 = jnp.array([1.5, 4.97, 5.5, 0.5, 0.09090909090909091])
w_indexes = {'M_F': 0, 'vs1': 1, 'rho': 2, 'alpha2': 3, 'alpha1': 4}

c = jnp.array([0.0, 1.0, 0.365, 7.1, 2.0, 0.2, 1.0, 1.0, 0.28, 1.0, 0.2, 2.0, 1.0, 1.0, 7.1, 15.0, 50.0, 5.0, 1.0, 1.0, 1.0]) 
c_indexes = {'kd5': 0, 'kd1': 1, 'V0': 2, 'Vsc': 3, 'n': 4, 'Ka': 5, 'kd3': 6, 'ks2': 7, 'kd2': 8, 'ks3': 9, 'Ki': 10, 'm': 11, 'kd4': 12, 'ks1': 13, 'RALDH2_0': 14, 'x': 15, 'L': 16, 'M_0': 17, 'Kr1': 18, 'Kr2': 19, 'PSM': 20}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.RA_synthesis(y, w, c, t), self.RA_decay(y, w, c, t), self.RA_deg_by_Cyp26(y, w, c, t), self.M_C_transcription(y, w, c, t), self.M_C_decay(y, w, c, t), self.C_translation(y, w, c, t), self.C_decay(y, w, c, t), self.FGF_synthesis(y, w, c, t), self.FGF_decay(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def RA_synthesis(self, y, w, c, t):
		return c[20] * w[1]


	def RA_decay(self, y, w, c, t):
		return c[20] * c[0] * (y[0]/1.0)


	def RA_deg_by_Cyp26(self, y, w, c, t):
		return c[20] * c[1] * (y[0]/1.0) * (y[2]/1.0)


	def M_C_transcription(self, y, w, c, t):
		return c[20] * (c[2] + c[3] * (y[3]/1.0)**c[4] / (c[5]**c[4] + (y[3]/1.0)**c[4]))


	def M_C_decay(self, y, w, c, t):
		return c[20] * c[6] * (y[1]/1.0)


	def C_translation(self, y, w, c, t):
		return c[20] * c[7] * (y[1]/1.0)


	def C_decay(self, y, w, c, t):
		return c[20] * c[8] * (y[2]/1.0)


	def FGF_synthesis(self, y, w, c, t):
		return c[20] * c[9] * (w[0]/1.0) * (c[10]**c[11] / (c[10]**c[11] + (y[0]/1.0)**c[11]))


	def FGF_decay(self, y, w, c, t):
		return c[20] * c[12] * (y[3]/1.0)

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(1.0 * (c[17] * (c[15] / c[16])))

		w = w.at[1].set((c[13] * c[14] * (1 - c[15] / c[16])))

		w = w.at[3].set(((y[3]/1.0) / ((y[3]/1.0) + c[19])))

		w = w.at[4].set(((y[0]/1.0) / ((y[0]/1.0) + c[18])))

		w = w.at[2].set((w[3] / w[4]))

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

	def __init__(self, y_indexes={'RA': 0, 'M_C': 1, 'C': 2, 'F': 3}, w_indexes={'M_F': 0, 'vs1': 1, 'rho': 2, 'alpha2': 3, 'alpha1': 4}, c_indexes={'kd5': 0, 'kd1': 1, 'V0': 2, 'Vsc': 3, 'n': 4, 'Ka': 5, 'kd3': 6, 'ks2': 7, 'kd2': 8, 'ks3': 9, 'Ki': 10, 'm': 11, 'kd4': 12, 'ks1': 13, 'RALDH2_0': 14, 'x': 15, 'L': 16, 'M_0': 17, 'Kr1': 18, 'Kr2': 19, 'PSM': 20}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.1, 0.1, 0.1, 1.0]), w0=jnp.array([1.5, 4.97, 5.5, 0.5, 0.09090909090909091]), c=jnp.array([0.0, 1.0, 0.365, 7.1, 2.0, 0.2, 1.0, 1.0, 0.28, 1.0, 0.2, 2.0, 1.0, 1.0, 7.1, 15.0, 50.0, 5.0, 1.0, 1.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

