import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.25, 0.0])
y_indexes = {'M': 0, 'C2': 1, 'YP': 2, 'CP': 3, 'Y': 4, 'pM': 5, 'EmptySet': 6}

w0 = jnp.array([0.25, 1.0])
w_indexes = {'YT': 0, 'CT': 1}

c = jnp.array([1.0, 1.0, 1000000.0, 1000.0, 200.0, 0.0, 0.015, 0.0, 0.6, 180.0, 0.018]) 
c_indexes = {'cell': 0, 'Reaction1_k6': 1, 'Reaction2_k8notP': 2, 'Reaction3_k9': 3, 'Reaction4_k3': 4, 'Reaction5_k5notP': 5, 'Reaction6_k1aa': 6, 'Reaction7_k2': 7, 'Reaction8_k7': 8, 'Reaction9_k4': 9, 'Reaction9_k4prime': 10}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.Reaction1(y, w, c, t), self.Reaction2(y, w, c, t), self.Reaction3(y, w, c, t), self.Reaction4(y, w, c, t), self.Reaction5(y, w, c, t), self.Reaction6(y, w, c, t), self.Reaction7(y, w, c, t), self.Reaction8(y, w, c, t), self.Reaction9(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def Reaction1(self, y, w, c, t):
		return c[0] * c[1] * (y[0]/1.0)


	def Reaction2(self, y, w, c, t):
		return c[0] * (y[1]/1.0) * c[2]


	def Reaction3(self, y, w, c, t):
		return c[0] * (y[3]/1.0) * c[3]


	def Reaction4(self, y, w, c, t):
		return c[0] * (y[3]/1.0) * c[4] * (y[4]/1.0)


	def Reaction5(self, y, w, c, t):
		return c[0] * c[5] * (y[0]/1.0)


	def Reaction6(self, y, w, c, t):
		return c[0] * c[6]


	def Reaction7(self, y, w, c, t):
		return c[0] * c[7] * (y[4]/1.0)


	def Reaction8(self, y, w, c, t):
		return c[0] * c[8] * (y[2]/1.0)


	def Reaction9(self, y, w, c, t):
		return c[0] * (y[5]/1.0) * (c[10] + c[9] * ((y[0]/1.0) / (w[1]/1.0))**2)

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(1.0 * ((y[4]/1.0) + (y[2]/1.0) + (y[0]/1.0) + (y[5]/1.0)))

		w = w.at[1].set(1.0 * ((y[1]/1.0) + (y[3]/1.0) + (y[0]/1.0) + (y[5]/1.0)))

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

	def __init__(self, y_indexes={'M': 0, 'C2': 1, 'YP': 2, 'CP': 3, 'Y': 4, 'pM': 5, 'EmptySet': 6}, w_indexes={'YT': 0, 'CT': 1}, c_indexes={'cell': 0, 'Reaction1_k6': 1, 'Reaction2_k8notP': 2, 'Reaction3_k9': 3, 'Reaction4_k3': 4, 'Reaction5_k5notP': 5, 'Reaction6_k1aa': 6, 'Reaction7_k2': 7, 'Reaction8_k7': 8, 'Reaction9_k4': 9, 'Reaction9_k4prime': 10}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.75, 0.0, 0.25, 0.0]), w0=jnp.array([0.25, 1.0]), c=jnp.array([1.0, 1.0, 1000000.0, 1000.0, 200.0, 0.0, 0.015, 0.0, 0.6, 180.0, 0.018]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

