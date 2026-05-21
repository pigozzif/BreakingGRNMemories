import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'ESA': 0, 'Pa': 1, 'ESB': 2, 'Pb': 3, 'Db': 4, 'BDb': 5, 'Da': 6, 'BDa': 7}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 1.0, 100.0, 100.0, 1e-05, 1e-05, 0.1, 0.1, 1.0]) 
c_indexes = {'da': 0, 'db': 1, 'sa': 2, 'sb': 3, 'ba': 4, 'bb': 5, 'ua': 6, 'ub': 7, 'default': 8}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, -2.0, 0.0, 0.0, 2.0], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, -2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.re2(y, w, c, t), self.re1(y, w, c, t), self.re12(y, w, c, t), self.re11(y, w, c, t), self.re13(y, w, c, t), self.re5(y, w, c, t), self.re7(y, w, c, t), self.re14(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def re2(self, y, w, c, t):
		return c[8] * c[2] * (y[6]/1.0)


	def re1(self, y, w, c, t):
		return c[8] * c[3] * (y[4]/1.0)


	def re12(self, y, w, c, t):
		return c[8] * c[0] * (y[1]/1.0)


	def re11(self, y, w, c, t):
		return c[8] * c[1] * (y[3]/1.0)


	def re13(self, y, w, c, t):
		return c[8] * c[5] * ((y[1]/1.0) * ((y[1]/1.0) - 1) / 2) * (y[4]/1.0)


	def re5(self, y, w, c, t):
		return c[8] * c[4] * ((y[3]/1.0) * ((y[3]/1.0) - 1) / 2) * (y[6]/1.0)


	def re7(self, y, w, c, t):
		return c[8] * c[6] * (y[7]/1.0)


	def re14(self, y, w, c, t):
		return c[8] * c[7] * (y[5]/1.0)

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

	def __init__(self, y_indexes={'ESA': 0, 'Pa': 1, 'ESB': 2, 'Pb': 3, 'Db': 4, 'BDb': 5, 'Da': 6, 'BDa': 7}, w_indexes={}, c_indexes={'da': 0, 'db': 1, 'sa': 2, 'sb': 3, 'ba': 4, 'bb': 5, 'ua': 6, 'ub': 7, 'default': 8}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 1.0, 100.0, 100.0, 1e-05, 1e-05, 0.1, 0.1, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

