import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([2800.0, 3.0, 0.0, 2.0, 900.0, 25.0, 0.0, 25.0, 20.0, 0.0, 20.0, 5.0, 0.0, 5.0, 500.0, 0.0, 50.0])
y_indexes = {'PEP': 0, 'EI': 1, 'PyrPI': 2, 'EIP': 3, 'Pyr': 4, 'HPr': 5, 'EIPHPr': 6, 'HPrP': 7, 'EIIA': 8, 'HPrPIIA': 9, 'EIIAP': 10, 'EIICB': 11, 'EIIAPIICB': 12, 'EIICBP': 13, 'Glc': 14, 'EIICBPGlc': 15, 'GlcP': 16}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 1960.0, 480000.0, 108000.0, 294.0, 14000.0, 14000.0, 84000.0, 3360.0, 21960.0, 21960.0, 4392.0, 3384.0, 880.0, 880.0, 2640.0, 960.0, 260.0, 389.0, 4800.0, 0.0054]) 
c_indexes = {'compartment': 0, 'v1_k1f': 1, 'v1_k1r': 2, 'v2_k2f': 3, 'v2_k2r': 4, 'v3_k3f': 5, 'v3_k3r': 6, 'v4_k4f': 7, 'v4_k4r': 8, 'v5_k5f': 9, 'v5_k5r': 10, 'v6_k6f': 11, 'v6_k6r': 12, 'v7_k7f': 13, 'v7_k7r': 14, 'v8_k8f': 15, 'v8_k8r': 16, 'v9_k9f': 17, 'v9_k9r': 18, 'v10_k10f': 19, 'v10_k10r': 20}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.v5(y, w, c, t), self.v6(y, w, c, t), self.v7(y, w, c, t), self.v8(y, w, c, t), self.v9(y, w, c, t), self.v10(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[0] * (c[1] * (y[0]/1.0) * (y[1]/1.0) - c[2] * (y[2]/1.0))


	def v2(self, y, w, c, t):
		return c[0] * (c[3] * (y[2]/1.0) - c[4] * (y[4]/1.0) * (y[3]/1.0))


	def v3(self, y, w, c, t):
		return c[0] * (c[5] * (y[3]/1.0) * (y[5]/1.0) - c[6] * (y[6]/1.0))


	def v4(self, y, w, c, t):
		return c[0] * (c[7] * (y[6]/1.0) - c[8] * (y[1]/1.0) * (y[7]/1.0))


	def v5(self, y, w, c, t):
		return c[0] * (c[9] * (y[7]/1.0) * (y[8]/1.0) - c[10] * (y[9]/1.0))


	def v6(self, y, w, c, t):
		return c[0] * (c[11] * (y[9]/1.0) - c[12] * (y[5]/1.0) * (y[10]/1.0))


	def v7(self, y, w, c, t):
		return c[0] * (c[13] * (y[10]/1.0) * (y[11]/1.0) - c[14] * (y[12]/1.0))


	def v8(self, y, w, c, t):
		return c[0] * (c[15] * (y[12]/1.0) - c[16] * (y[8]/1.0) * (y[13]/1.0))


	def v9(self, y, w, c, t):
		return c[0] * (c[17] * (y[13]/1.0) * (y[14]/1.0) - c[18] * (y[15]/1.0))


	def v10(self, y, w, c, t):
		return c[0] * (c[19] * (y[15]/1.0) - c[20] * (y[11]/1.0) * (y[16]/1.0))

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

	def __init__(self, y_indexes={'PEP': 0, 'EI': 1, 'PyrPI': 2, 'EIP': 3, 'Pyr': 4, 'HPr': 5, 'EIPHPr': 6, 'HPrP': 7, 'EIIA': 8, 'HPrPIIA': 9, 'EIIAP': 10, 'EIICB': 11, 'EIIAPIICB': 12, 'EIICBP': 13, 'Glc': 14, 'EIICBPGlc': 15, 'GlcP': 16}, w_indexes={}, c_indexes={'compartment': 0, 'v1_k1f': 1, 'v1_k1r': 2, 'v2_k2f': 3, 'v2_k2r': 4, 'v3_k3f': 5, 'v3_k3r': 6, 'v4_k4f': 7, 'v4_k4r': 8, 'v5_k5f': 9, 'v5_k5r': 10, 'v6_k6f': 11, 'v6_k6r': 12, 'v7_k7f': 13, 'v7_k7r': 14, 'v8_k8f': 15, 'v8_k8r': 16, 'v9_k9f': 17, 'v9_k9r': 18, 'v10_k10f': 19, 'v10_k10r': 20}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([2800.0, 3.0, 0.0, 2.0, 900.0, 25.0, 0.0, 25.0, 20.0, 0.0, 20.0, 5.0, 0.0, 5.0, 500.0, 0.0, 50.0]), w0=jnp.array([]), c=jnp.array([1.0, 1960.0, 480000.0, 108000.0, 294.0, 14000.0, 14000.0, 84000.0, 3360.0, 21960.0, 21960.0, 4392.0, 3384.0, 880.0, 880.0, 2640.0, 960.0, 260.0, 389.0, 4800.0, 0.0054]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

