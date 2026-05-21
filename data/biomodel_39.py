import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.76, 0.35, 0.29, 85.45, 34.55])
y_indexes = {'CaER': 0, 'Ca_cyt': 1, 'CaM': 2, 'CaPr': 3, 'Pr': 4}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 1.0, 1.0, 4100.0, 5.0, 0.05, 20.0, 125.0, 5.0, 0.00625, 300.0, 0.8, 0.01, 0.1]) 
c_indexes = {'Cytosol': 0, 'Endoplasmic_Reticulum': 1, 'Mitochondria': 2, 'v1_Kch': 3, 'v1_K1': 4, 'v3_Kleak': 5, 'v5_Kpump': 6, 'v7_Kout': 7, 'v7_K3': 8, 'v7_Km': 9, 'v9_Kin': 10, 'v9_K2': 11, 'v11_Kminus': 12, 'v12_Kplus': 13}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-0.25, -0.25, 0.25, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, -0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v3(y, w, c, t), self.v5(y, w, c, t), self.v7(y, w, c, t), self.v9(y, w, c, t), self.v11(y, w, c, t), self.v12(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[0] * (c[3] * (y[1]/1.0)**2 * ((y[0]/1.0) - (y[1]/1.0)) / (c[4]**2 + (y[1]/1.0)**2))


	def v3(self, y, w, c, t):
		return c[0] * c[5] * ((y[0]/1.0) - (y[1]/1.0))


	def v5(self, y, w, c, t):
		return c[1] * c[6] * (y[1]/1.0)


	def v7(self, y, w, c, t):
		return c[0] * (y[2]/1.0) * (c[7] * (y[1]/1.0)**2 / (c[8]**2 + (y[1]/1.0)**2) + c[9])


	def v9(self, y, w, c, t):
		return c[2] * (c[10] * (y[1]/1.0)**8 / (c[11]**8 + (y[1]/1.0)**8))


	def v11(self, y, w, c, t):
		return c[0] * c[12] * (y[3]/1.0)


	def v12(self, y, w, c, t):
		return c[0] * c[13] * (y[1]/1.0) * (y[4]/1.0)

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

	def __init__(self, y_indexes={'CaER': 0, 'Ca_cyt': 1, 'CaM': 2, 'CaPr': 3, 'Pr': 4}, w_indexes={}, c_indexes={'Cytosol': 0, 'Endoplasmic_Reticulum': 1, 'Mitochondria': 2, 'v1_Kch': 3, 'v1_K1': 4, 'v3_Kleak': 5, 'v5_Kpump': 6, 'v7_Kout': 7, 'v7_K3': 8, 'v7_Km': 9, 'v9_Kin': 10, 'v9_K2': 11, 'v11_Kminus': 12, 'v12_Kplus': 13}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.76, 0.35, 0.29, 85.45, 34.55]), w0=jnp.array([]), c=jnp.array([1.0, 1.0, 1.0, 4100.0, 5.0, 0.05, 20.0, 125.0, 5.0, 0.00625, 300.0, 0.8, 0.01, 0.1]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

