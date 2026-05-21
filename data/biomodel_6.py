import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0, 0.0, 0.0, 0.0])
y_indexes = {'EmptySet': 0, 'z': 1, 'u': 2, 'v': 3}

w0 = jnp.array([0.0, 9.999999999999999e-05])
w_indexes = {'z': 0, 'alpha': 1}

c = jnp.array([0.015, 1.0, 180.0, 0.018, 1.0]) 
c_indexes = {'kappa': 0, 'k6': 1, 'k4': 2, 'k4prime': 3, 'cell': 4}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, self.Rateu(y, w, c, t), self.Ratev(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.Reaction1(y, w, c, t), self.Reaction2(y, w, c, t), self.Reaction3(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def Reaction1(self, y, w, c, t):
		return c[0]


	def Reaction2(self, y, w, c, t):
		return c[1] * y[2]


	def Reaction3(self, y, w, c, t):
		return c[2] * w[0] * (c[3] / c[2] + y[2]**2)

	def Rateu(self, y, w, c, t):
		return c[2] * (y[3] - y[2]) * (w[1] + y[2]**2) - c[1] * y[2]

	def Ratev(self, y, w, c, t):
		return c[0] - c[1] * y[2]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set((y[3] - y[2]))

		w = w.at[1].set((c[3] / c[2]))

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

	def __init__(self, y_indexes={'EmptySet': 0, 'z': 1, 'u': 2, 'v': 3}, w_indexes={'z': 0, 'alpha': 1}, c_indexes={'kappa': 0, 'k6': 1, 'k4': 2, 'k4prime': 3, 'cell': 4}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([1.0, 0.0, 0.0, 0.0]), w0=jnp.array([0.0, 9.999999999999999e-05]), c=jnp.array([0.015, 1.0, 180.0, 0.018, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

