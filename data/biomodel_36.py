import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0, 1.0])
y_indexes = {'M': 0, 'P': 1}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 6.0221367e+23, 1.0, 1.0, 1.0, 0.1, 200.0, 0.5, 0.1, 0.1, 10.0, 0.03, 0.05, 200.0]) 
c_indexes = {'EmptySet': 0, 'N_A': 1, 'default': 2, 'CYTOPLASM': 3, 'TC_Vm': 4, 'TC_Pcrit': 5, 'TC_Keq': 6, 'TL_V': 7, 'mRNAD_D': 8, 'ProteinD_D': 9, 'DBT_k1': 10, 'DBT_k2': 11, 'DBT_J': 12, 'DBT_Keq': 13}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.TC(y, w, c, t), self.TL(y, w, c, t), self.mRNAD(y, w, c, t), self.ProteinD(y, w, c, t), self.DBT(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def TC(self, y, w, c, t):
		return c[3] * (c[4] / (1 + ((y[1]/1.0) * (1 - 2 / (1 + (1 + 8 * c[6] * (y[1]/1.0))**0.5)) / (2 * c[5]))**2))


	def TL(self, y, w, c, t):
		return c[7] * (y[0]/1.0) * c[3]


	def mRNAD(self, y, w, c, t):
		return c[8] * (y[0]/1.0) * c[3]


	def ProteinD(self, y, w, c, t):
		return c[9] * (y[1]/1.0) * c[3]


	def DBT(self, y, w, c, t):
		return c[3] * ((c[10] * (y[1]/1.0) * (2 / (1 + (1 + 8 * c[13] * (y[1]/1.0))**0.5)) + c[11] * (y[1]/1.0)) / (c[12] + (y[1]/1.0)))

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

	def __init__(self, y_indexes={'M': 0, 'P': 1}, w_indexes={}, c_indexes={'EmptySet': 0, 'N_A': 1, 'default': 2, 'CYTOPLASM': 3, 'TC_Vm': 4, 'TC_Pcrit': 5, 'TC_Keq': 6, 'TL_V': 7, 'mRNAD_D': 8, 'ProteinD_D': 9, 'DBT_k1': 10, 'DBT_k2': 11, 'DBT_J': 12, 'DBT_Keq': 13}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([1.0, 1.0]), w0=jnp.array([]), c=jnp.array([0.0, 6.0221367e+23, 1.0, 1.0, 1.0, 0.1, 200.0, 0.5, 0.1, 0.1, 10.0, 0.03, 0.05, 200.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

