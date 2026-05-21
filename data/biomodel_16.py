import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0000000000000001e-16, 2.5e-16, 2.5e-16, 2.5e-16, 2.5e-16])
y_indexes = {'M': 0, 'P0': 1, 'P1': 2, 'P2': 3, 'Pn': 4}

w0 = jnp.array([1e-15])
w_indexes = {'Pt': 0}

c = jnp.array([0.0, 1e-15, 1e-15, 1e-15, 0.76, 1.0, 4.0, 0.38, 3.2, 2.0, 1.58, 2.0, 5.0, 2.0, 2.5, 2.0, 1.9, 1.3, 0.5, 0.65, 0.95, 0.2]) 
c_indexes = {'EmptySet': 0, 'default': 1, 'CYTOPLASM': 2, 'compartment_0000004': 3, 'rM_Vs': 4, 'rM_KI': 5, 'rM_n': 6, 'rTL_ks': 7, 'rP01_V1': 8, 'rP01_K1': 9, 'rP10_V2': 10, 'rP10_K2': 11, 'rP12_V3': 12, 'rP12_K3': 13, 'rP21_V4': 14, 'rP21_K4': 15, 'rP2n_k1': 16, 'rPn2_k2': 17, 'rmRNAd_Km': 18, 'rmRNAd_Vm': 19, 'rVd_Vd': 20, 'rVd_Kd': 21}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.rM(y, w, c, t), self.rTL(y, w, c, t), self.rP01(y, w, c, t), self.rP10(y, w, c, t), self.rP12(y, w, c, t), self.rP21(y, w, c, t), self.rP2n(y, w, c, t), self.rPn2(y, w, c, t), self.rmRNAd(y, w, c, t), self.rVd(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def rM(self, y, w, c, t):
		return c[1] * c[4] * c[5]**c[6] / (c[5]**c[6] + (y[4]/1e-15)**c[6])


	def rTL(self, y, w, c, t):
		return c[7] * (y[0]/1e-15) * c[1]


	def rP01(self, y, w, c, t):
		return c[2] * c[8] * (y[1]/1e-15) / (c[9] + (y[1]/1e-15))


	def rP10(self, y, w, c, t):
		return c[2] * c[10] * (y[2]/1e-15) / (c[11] + (y[2]/1e-15))


	def rP12(self, y, w, c, t):
		return c[2] * c[12] * (y[2]/1e-15) / (c[13] + (y[2]/1e-15))


	def rP21(self, y, w, c, t):
		return c[2] * c[14] * (y[3]/1e-15) / (c[15] + (y[3]/1e-15))


	def rP2n(self, y, w, c, t):
		return c[16] * (y[3]/1e-15) * c[2]


	def rPn2(self, y, w, c, t):
		return c[17] * (y[4]/1e-15) * c[3]


	def rmRNAd(self, y, w, c, t):
		return c[19] * (y[0]/1e-15) * c[2] / (c[18] + (y[0]/1e-15))


	def rVd(self, y, w, c, t):
		return c[2] * c[20] * (y[3]/1e-15) / (c[21] + (y[3]/1e-15))

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(1e-15 * ((y[1]/1e-15) + (y[2]/1e-15) + (y[3]/1e-15) + (y[4]/1e-15)))

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

	def __init__(self, y_indexes={'M': 0, 'P0': 1, 'P1': 2, 'P2': 3, 'Pn': 4}, w_indexes={'Pt': 0}, c_indexes={'EmptySet': 0, 'default': 1, 'CYTOPLASM': 2, 'compartment_0000004': 3, 'rM_Vs': 4, 'rM_KI': 5, 'rM_n': 6, 'rTL_ks': 7, 'rP01_V1': 8, 'rP01_K1': 9, 'rP10_V2': 10, 'rP10_K2': 11, 'rP12_V3': 12, 'rP12_K3': 13, 'rP21_V4': 14, 'rP21_K4': 15, 'rP2n_k1': 16, 'rPn2_k2': 17, 'rmRNAd_Km': 18, 'rmRNAd_Vm': 19, 'rVd_Vd': 20, 'rVd_Kd': 21}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([1.0000000000000001e-16, 2.5e-16, 2.5e-16, 2.5e-16, 2.5e-16]), w0=jnp.array([1e-15]), c=jnp.array([0.0, 1e-15, 1e-15, 1e-15, 0.76, 1.0, 4.0, 0.38, 3.2, 2.0, 1.58, 2.0, 5.0, 2.0, 2.5, 2.0, 1.9, 1.3, 0.5, 0.65, 0.95, 0.2]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

