import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01])
y_indexes = {'OCT4_Gene': 0, 'OCT4': 1, 'degradation': 2, 'SOX2_Gene': 3, 'SOX2': 4, 'NANOG_Gene': 5, 'NANOG': 6, 'CDX2_Gene': 7, 'CDX2': 8, 'GCNF_Gene': 9, 'GCNF': 10, 'GATA6_Gene': 11, 'GATA6': 12}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([0.01, 0.0, 10.0, 0.1, 0.0, 0.001, 0.02, 0.0125, 0.025, 1.0, 0.02, 0.0125, 0.03, 10.0, 10.0, 0.1, 0.001, 0.05, 0.0125, 0.001, 0.05, 0.0125, 0.05, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1, 0.1, 10.0, 0.1, 0.001, 2.0, 2.0, 5.0, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.00025, 1.0, 0.00025, 15.0, 0.01, 0.01, 1.0]) 
c_indexes = {'targetGene': 0, 'p53': 1, 'A': 2, 'OCT4_SOX2': 3, 'Protein': 4, 'a0': 5, 'a1': 6, 'a2': 7, 'a3': 8, 'b0': 9, 'b1': 10, 'b2': 11, 'b3': 12, 'b4': 13, 'b5': 14, 'gamma1': 15, 'c0': 16, 'c1': 17, 'c2': 18, 'd0': 19, 'd1': 20, 'd2': 21, 'd3': 22, 'gamma2': 23, 'e0': 24, 'e1': 25, 'e2': 26, 'f0': 27, 'f1': 28, 'f2': 29, 'f3': 30, 'gamma3': 31, 'g0': 32, 'g1': 33, 'h0': 34, 'h1': 35, 'gamma4': 36, 'i0': 37, 'i1': 38, 'i2': 39, 'j0': 40, 'j1': 41, 'gamma5': 42, 'p0': 43, 'p1': 44, 'p2': 45, 'q0': 46, 'q1': 47, 'q2': 48, 'gammag': 49, 'gamman': 50, 'cell': 51}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.R1(y, w, c, t), self.R2(y, w, c, t), self.R3(y, w, c, t), self.R4(y, w, c, t), self.R5(y, w, c, t), self.R6(y, w, c, t), self.R7(y, w, c, t), self.R8(y, w, c, t), self.R9(y, w, c, t), self.R10(y, w, c, t), self.R11(y, w, c, t), self.R12(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def R1(self, y, w, c, t):
		return (c[5] + c[6] * (c[2]/1.0) + c[7] * (y[1]/1.0) * (y[4]/1.0) + c[8] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0)) / (1 + c[9] * (c[2]/1.0) + c[10] * (y[1]/1.0) + c[11] * (y[1]/1.0) * (y[4]/1.0) + c[12] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0) + c[13] * (y[8]/1.0) * (y[1]/1.0) + c[14] * (y[10]/1.0))


	def R2(self, y, w, c, t):
		return c[15] * (y[1]/1.0)


	def R3(self, y, w, c, t):
		return (c[16] + c[17] * (y[1]/1.0) * (y[4]/1.0) + c[18] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0)) / (1 + c[19] * (y[1]/1.0) + c[20] * (y[1]/1.0) * (y[4]/1.0) + c[21] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0))


	def R4(self, y, w, c, t):
		return c[23] * (y[4]/1.0)


	def R5(self, y, w, c, t):
		return (c[6] * (c[3]/1.0) + c[7] * (c[3]/1.0) * (y[6]/1.0)) / (1 + c[10] * (c[3]/1.0) + c[11] * (c[3]/1.0) * (y[6]/1.0) + c[12] * (c[3]/1.0) * (y[12]/1.0))


	def R6(self, y, w, c, t):
		return c[50] * (y[6]/1.0)


	def R7(self, y, w, c, t):
		return (c[32] + c[33] * (y[8]/1.0)) / (1 + c[34] * (y[8]/1.0) + c[35] * (y[8]/1.0) * (y[1]/1.0))


	def R8(self, y, w, c, t):
		return c[36] * (y[8]/1.0)


	def R9(self, y, w, c, t):
		return (c[37] + c[38] * (y[8]/1.0) + c[39] * (y[12]/1.0)) / (1 + c[40] * (y[8]/1.0) + c[41] * (y[12]/1.0))


	def R10(self, y, w, c, t):
		return c[42] * (y[10]/1.0)


	def R11(self, y, w, c, t):
		return (c[17] * (c[3]/1.0) + c[18] * (y[12]/1.0)) / (1 + c[20] * (c[3]/1.0) + c[21] * (y[12]/1.0) + c[22] * (y[6]/1.0))


	def R12(self, y, w, c, t):
		return c[49] * (y[12]/1.0)

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

	def __init__(self, y_indexes={'OCT4_Gene': 0, 'OCT4': 1, 'degradation': 2, 'SOX2_Gene': 3, 'SOX2': 4, 'NANOG_Gene': 5, 'NANOG': 6, 'CDX2_Gene': 7, 'CDX2': 8, 'GCNF_Gene': 9, 'GCNF': 10, 'GATA6_Gene': 11, 'GATA6': 12}, w_indexes={}, c_indexes={'targetGene': 0, 'p53': 1, 'A': 2, 'OCT4_SOX2': 3, 'Protein': 4, 'a0': 5, 'a1': 6, 'a2': 7, 'a3': 8, 'b0': 9, 'b1': 10, 'b2': 11, 'b3': 12, 'b4': 13, 'b5': 14, 'gamma1': 15, 'c0': 16, 'c1': 17, 'c2': 18, 'd0': 19, 'd1': 20, 'd2': 21, 'd3': 22, 'gamma2': 23, 'e0': 24, 'e1': 25, 'e2': 26, 'f0': 27, 'f1': 28, 'f2': 29, 'f3': 30, 'gamma3': 31, 'g0': 32, 'g1': 33, 'h0': 34, 'h1': 35, 'gamma4': 36, 'i0': 37, 'i1': 38, 'i2': 39, 'j0': 40, 'j1': 41, 'gamma5': 42, 'p0': 43, 'p1': 44, 'p2': 45, 'q0': 46, 'q1': 47, 'q2': 48, 'gammag': 49, 'gamman': 50, 'cell': 51}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01]), w0=jnp.array([]), c=jnp.array([0.01, 0.0, 10.0, 0.1, 0.0, 0.001, 0.02, 0.0125, 0.025, 1.0, 0.02, 0.0125, 0.03, 10.0, 10.0, 0.1, 0.001, 0.05, 0.0125, 0.001, 0.05, 0.0125, 0.05, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1, 0.1, 10.0, 0.1, 0.001, 2.0, 2.0, 5.0, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.00025, 1.0, 0.00025, 15.0, 0.01, 0.01, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

