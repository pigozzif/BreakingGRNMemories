import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.1, 0.0, 0.01, 0.0])
y_indexes = {'OCT4_Gene': 0, 'OCT4': 1, 'degradation': 2, 'NANOG_Gene': 3, 'NANOG': 4, 'SOX2': 5, 'OCT4_SOX2': 6, 'SOX2_Gene': 7, 'targetGene': 8, 'Protein': 9}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.0, 10.0, 0.0001, 1.0, 0.01, 0.2, 1000.0, 0.0011, 0.001, 0.0007, 1.0, 0.0001, 0.005, 0.1, 0.000995, 0.001, 0.01, 1.0, 0.05, 0.001, 5.0, 0.0001, 1.0, 0.01, 0.2, 0.0011, 0.001, 0.0007, 1.0, 0.1, 0.0001, 0.0019, 0.05, 0.01, 1.0]) 
c_indexes = {'p53': 0, 'A': 1, 'eta1': 2, 'a1': 3, 'a2': 4, 'a3': 5, 'f': 6, 'b1': 7, 'b2': 8, 'b3': 9, 'gamma1': 10, 'eta5': 11, 'e1': 12, 'e2': 13, 'f2': 14, 'f1': 15, 'f3': 16, 'gamma2': 17, 'k1c': 18, 'k2c': 19, 'k3c': 20, 'eta3': 21, 'c1': 22, 'c2': 23, 'c3': 24, 'd1': 25, 'd2': 26, 'd3': 27, 'gamma3': 28, 'g1': 29, 'eta7': 30, 'h1': 31, 'h2': 32, 'gamma4': 33, 'compartment': 34}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.J0(y, w, c, t), self.J1(y, w, c, t), self.J2(y, w, c, t), self.J3(y, w, c, t), self.J4(y, w, c, t), self.J5(y, w, c, t), self.J6(y, w, c, t), self.J7(y, w, c, t), self.J8(y, w, c, t), self.J9(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def J0(self, y, w, c, t):
		return (c[2] + c[3] * (c[1]/1.0) + c[4] * (y[6]/1.0) + c[5] * (y[6]/1.0) * (y[4]/1.0)) / (1 + c[2] / c[6] + c[7] * (c[1]/1.0) + c[8] * (y[6]/1.0) + c[9] * (y[6]/1.0) * (y[4]/1.0))


	def J1(self, y, w, c, t):
		return c[10] * (y[1]/1.0)


	def J2(self, y, w, c, t):
		return (c[11] + c[12] * (y[6]/1.0) + c[13] * (y[6]/1.0) * (y[4]/1.0)) / (1 + c[11] / c[6] + c[14] * (y[6]/1.0) + c[15] * (y[6]/1.0) * (y[4]/1.0) + c[16] * (c[0]/1.0))


	def J3(self, y, w, c, t):
		return c[17] * (y[4]/1.0)


	def J4(self, y, w, c, t):
		return c[18] * (y[1]/1.0) * (y[5]/1.0) - c[19] * (y[6]/1.0)


	def J5(self, y, w, c, t):
		return c[20] * (y[6]/1.0)


	def J6(self, y, w, c, t):
		return (c[21] + c[22] * (c[1]/1.0) + c[23] * (y[6]/1.0) + c[24] * (y[6]/1.0) * (y[4]/1.0)) / (1 + c[21] / c[6] + c[25] * (c[1]/1.0) + c[26] * (y[6]/1.0) + c[27] * (y[6]/1.0) * (y[4]/1.0))


	def J7(self, y, w, c, t):
		return c[28] * (y[5]/1.0)


	def J8(self, y, w, c, t):
		return (c[29] * (y[6]/1.0) + c[30]) / (1 + c[30] / c[6] + c[31] * (y[6]/1.0) + c[32] * (y[6]/1.0) * (y[4]/1.0))


	def J9(self, y, w, c, t):
		return c[33] * (y[9]/1.0)

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

	def __init__(self, y_indexes={'OCT4_Gene': 0, 'OCT4': 1, 'degradation': 2, 'NANOG_Gene': 3, 'NANOG': 4, 'SOX2': 5, 'OCT4_SOX2': 6, 'SOX2_Gene': 7, 'targetGene': 8, 'Protein': 9}, w_indexes={}, c_indexes={'p53': 0, 'A': 1, 'eta1': 2, 'a1': 3, 'a2': 4, 'a3': 5, 'f': 6, 'b1': 7, 'b2': 8, 'b3': 9, 'gamma1': 10, 'eta5': 11, 'e1': 12, 'e2': 13, 'f2': 14, 'f1': 15, 'f3': 16, 'gamma2': 17, 'k1c': 18, 'k2c': 19, 'k3c': 20, 'eta3': 21, 'c1': 22, 'c2': 23, 'c3': 24, 'd1': 25, 'd2': 26, 'd3': 27, 'gamma3': 28, 'g1': 29, 'eta7': 30, 'h1': 31, 'h2': 32, 'gamma4': 33, 'compartment': 34}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.1, 0.0, 0.01, 0.0]), w0=jnp.array([]), c=jnp.array([0.0, 10.0, 0.0001, 1.0, 0.01, 0.2, 1000.0, 0.0011, 0.001, 0.0007, 1.0, 0.0001, 0.005, 0.1, 0.000995, 0.001, 0.01, 1.0, 0.05, 0.001, 5.0, 0.0001, 1.0, 0.01, 0.2, 0.0011, 0.001, 0.0007, 1.0, 0.1, 0.0001, 0.0019, 0.05, 0.01, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

