import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([500.0, 50.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'M': 0, 'MAPKK': 1, 'M_MAPKK': 2, 'Mp': 3, 'Mp_MAPKK': 4, 'Mpp': 5, 'MKP3': 6, 'Mpp_MKP3': 7, 'Mp_MKP3_dep': 8, 'Mp_MKP3': 9, 'M_MKP3': 10}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.02, 1.0, 0.01, 0.032, 1.0, 15.0, 0.045, 1.0, 0.092, 1.0, 0.01, 0.01, 1.0, 0.5, 0.086, 0.0011, 1.0]) 
c_indexes = {'k1': 0, 'k_1': 1, 'k2': 2, 'k3': 3, 'k_3': 4, 'k4': 5, 'h1': 6, 'h_1': 7, 'h2': 8, 'h3': 9, 'h_3': 10, 'h4': 11, 'h_4': 12, 'h5': 13, 'h6': 14, 'h_6': 15, 'uVol': 16}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1a(y, w, c, t), self.v1b(y, w, c, t), self.v2a(y, w, c, t), self.v2b(y, w, c, t), self.v3a(y, w, c, t), self.v3b(y, w, c, t), self.v3c(y, w, c, t), self.v4a(y, w, c, t), self.v4b(y, w, c, t), self.v4c(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1a(self, y, w, c, t):
		return c[16] * (c[0] * (y[0]/1.0) * (y[1]/1.0) - c[1] * (y[2]/1.0))


	def v1b(self, y, w, c, t):
		return c[16] * c[2] * (y[2]/1.0)


	def v2a(self, y, w, c, t):
		return c[16] * (c[3] * (y[3]/1.0) * (y[1]/1.0) - c[4] * (y[4]/1.0))


	def v2b(self, y, w, c, t):
		return c[16] * c[5] * (y[4]/1.0)


	def v3a(self, y, w, c, t):
		return c[16] * (c[6] * (y[5]/1.0) * (y[6]/1.0) - c[7] * (y[7]/1.0))


	def v3b(self, y, w, c, t):
		return c[16] * c[8] * (y[7]/1.0)


	def v3c(self, y, w, c, t):
		return c[9] * (y[8]/1.0) - c[10] * (y[3]/1.0) * (y[6]/1.0)


	def v4a(self, y, w, c, t):
		return c[16] * (c[11] * (y[3]/1.0) * (y[6]/1.0) - c[12] * (y[9]/1.0))


	def v4b(self, y, w, c, t):
		return c[16] * c[13] * (y[9]/1.0)


	def v4c(self, y, w, c, t):
		return c[16] * (c[14] * (y[10]/1.0) - c[15] * (y[0]/1.0) * (y[6]/1.0))

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

	def __init__(self, y_indexes={'M': 0, 'MAPKK': 1, 'M_MAPKK': 2, 'Mp': 3, 'Mp_MAPKK': 4, 'Mpp': 5, 'MKP3': 6, 'Mpp_MKP3': 7, 'Mp_MKP3_dep': 8, 'Mp_MKP3': 9, 'M_MKP3': 10}, w_indexes={}, c_indexes={'k1': 0, 'k_1': 1, 'k2': 2, 'k3': 3, 'k_3': 4, 'k4': 5, 'h1': 6, 'h_1': 7, 'h2': 8, 'h3': 9, 'h_3': 10, 'h4': 11, 'h_4': 12, 'h5': 13, 'h6': 14, 'h_6': 15, 'uVol': 16}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([500.0, 50.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([0.02, 1.0, 0.01, 0.032, 1.0, 15.0, 0.045, 1.0, 0.092, 1.0, 0.01, 0.01, 1.0, 0.5, 0.086, 0.0011, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

