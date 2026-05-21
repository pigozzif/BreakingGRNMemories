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

c = jnp.array([0.01, 0.0, 25.0, 0.0, 0.0, 0.1, 0.0, 0.001, 1.0, 0.005, 0.025, 1.0, 0.001, 0.005, 0.025, 10.0, 10.0, 0.1, 0.001, 0.005, 0.025, 0.001, 0.005, 0.025, 0.05, 0.1, 0.001, 0.1, 0.1, 1.0, 0.001, 0.1, 0.1, 10.0, 1.0, 0.1, 0.001, 2.0, 2.0, 5.0, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.00025, 1.0, 0.00025, 15.0, 10.0, 0.1, 0.01, 1.0]) 
c_indexes = {'targetGene': 0, 'p53': 1, 'A': 2, 'SG': 3, 'SN': 4, 'OCT4_SOX2': 5, 'Protein': 6, 'a0': 7, 'a1': 8, 'a2': 9, 'a3': 10, 'b0': 11, 'b1': 12, 'b2': 13, 'b3': 14, 'b4': 15, 'b5': 16, 'gamma1': 17, 'c0': 18, 'c1': 19, 'c2': 20, 'd0': 21, 'd1': 22, 'd2': 23, 'd3': 24, 'gamma2': 25, 'e0': 26, 'e1': 27, 'e2': 28, 'e3': 29, 'f0': 30, 'f1': 31, 'f2': 32, 'f3': 33, 'f4': 34, 'gamma3': 35, 'g0': 36, 'g1': 37, 'h0': 38, 'h1': 39, 'gamma4': 40, 'i0': 41, 'i1': 42, 'i2': 43, 'j0': 44, 'j1': 45, 'gamma5': 46, 'p0': 47, 'p1': 48, 'p2': 49, 'q0': 50, 'q1': 51, 'q2': 52, 'q3': 53, 'gammag': 54, 'gamman': 55, 'cell': 56}

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
		return (c[7] + c[8] * (c[2]/1.0) + c[9] * (y[1]/1.0) * (y[4]/1.0) + c[10] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0)) / (1 + c[11] * (c[2]/1.0) + c[12] * (y[1]/1.0) + c[13] * (y[1]/1.0) * (y[4]/1.0) + c[14] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0) + c[15] * (y[8]/1.0) * (y[1]/1.0) + c[16] * (y[10]/1.0))


	def R2(self, y, w, c, t):
		return c[17] * (y[1]/1.0)


	def R3(self, y, w, c, t):
		return (c[18] + c[19] * (y[1]/1.0) * (y[4]/1.0) + c[20] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0)) / (1 + c[21] * (y[1]/1.0) + c[22] * (y[1]/1.0) * (y[4]/1.0) + c[23] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0))


	def R4(self, y, w, c, t):
		return c[25] * (y[4]/1.0)


	def R5(self, y, w, c, t):
		return (c[26] + c[27] * (y[1]/1.0) * (y[4]/1.0) + c[28] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0) + c[29] * (c[4]/1.0)) / (1 + c[30] * (y[1]/1.0) + c[31] * (y[1]/1.0) * (y[4]/1.0) + c[32] * (y[1]/1.0) * (y[4]/1.0) * (y[6]/1.0) + c[33] * (y[1]/1.0) * (y[12]/1.0) + c[34] * (c[4]/1.0))


	def R6(self, y, w, c, t):
		return c[35] * (y[6]/1.0)


	def R7(self, y, w, c, t):
		return (c[36] + c[37] * (y[8]/1.0)) / (1 + c[38] * (y[8]/1.0) + c[39] * (y[8]/1.0) * (y[1]/1.0))


	def R8(self, y, w, c, t):
		return c[40] * (y[8]/1.0)


	def R9(self, y, w, c, t):
		return (c[41] + c[42] * (y[8]/1.0) + c[43] * (y[12]/1.0)) / (1 + c[44] * (y[8]/1.0) + c[45] * (y[12]/1.0))


	def R10(self, y, w, c, t):
		return c[46] * (y[10]/1.0)


	def R11(self, y, w, c, t):
		return (c[47] + c[48] * (y[1]/1.0) + c[49] * (y[12]/1.0)) / (1 + c[50] * (y[1]/1.0) + c[51] * (y[12]/1.0) + c[52] * (y[6]/1.0) + c[53] * (c[3]/1.0))


	def R12(self, y, w, c, t):
		return c[54] * (y[12]/1.0)

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

	def __init__(self, y_indexes={'OCT4_Gene': 0, 'OCT4': 1, 'degradation': 2, 'SOX2_Gene': 3, 'SOX2': 4, 'NANOG_Gene': 5, 'NANOG': 6, 'CDX2_Gene': 7, 'CDX2': 8, 'GCNF_Gene': 9, 'GCNF': 10, 'GATA6_Gene': 11, 'GATA6': 12}, w_indexes={}, c_indexes={'targetGene': 0, 'p53': 1, 'A': 2, 'SG': 3, 'SN': 4, 'OCT4_SOX2': 5, 'Protein': 6, 'a0': 7, 'a1': 8, 'a2': 9, 'a3': 10, 'b0': 11, 'b1': 12, 'b2': 13, 'b3': 14, 'b4': 15, 'b5': 16, 'gamma1': 17, 'c0': 18, 'c1': 19, 'c2': 20, 'd0': 21, 'd1': 22, 'd2': 23, 'd3': 24, 'gamma2': 25, 'e0': 26, 'e1': 27, 'e2': 28, 'e3': 29, 'f0': 30, 'f1': 31, 'f2': 32, 'f3': 33, 'f4': 34, 'gamma3': 35, 'g0': 36, 'g1': 37, 'h0': 38, 'h1': 39, 'gamma4': 40, 'i0': 41, 'i1': 42, 'i2': 43, 'j0': 44, 'j1': 45, 'gamma5': 46, 'p0': 47, 'p1': 48, 'p2': 49, 'q0': 50, 'q1': 51, 'q2': 52, 'q3': 53, 'gammag': 54, 'gamman': 55, 'cell': 56}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01]), w0=jnp.array([]), c=jnp.array([0.01, 0.0, 25.0, 0.0, 0.0, 0.1, 0.0, 0.001, 1.0, 0.005, 0.025, 1.0, 0.001, 0.005, 0.025, 10.0, 10.0, 0.1, 0.001, 0.005, 0.025, 0.001, 0.005, 0.025, 0.05, 0.1, 0.001, 0.1, 0.1, 1.0, 0.001, 0.1, 0.1, 10.0, 1.0, 0.1, 0.001, 2.0, 2.0, 5.0, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.00025, 1.0, 0.00025, 15.0, 10.0, 0.1, 0.01, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

