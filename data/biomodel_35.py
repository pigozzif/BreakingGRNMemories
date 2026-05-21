import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
y_indexes = {'A': 0, 'R': 1, 'C': 2, 'EmptySet': 3, 'DA': 4, 'DAp': 5, 'MA': 6, 'DR': 7, 'DRp': 8, 'MR': 9}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 2.0, 1.0, 1.0, 0.2, 1.0, 50.0, 50.0, 500.0, 10.0, 50.0, 1.0, 100.0, 0.01, 50.0, 0.5, 5.0]) 
c_indexes = {'deterministicOscillator': 0, 'Reaction1_gammaC': 1, 'Reaction2_deltaA': 2, 'Reaction3_deltaA': 3, 'Reaction4_deltaR': 4, 'Reaction5_gammaA': 5, 'Reaction6_thetaA': 6, 'Reaction7_alphaA': 7, 'Reaction8_alphaAp': 8, 'Reaction9_deltaMA': 9, 'Reaction10_betaA': 10, 'Reaction11_gammaR': 11, 'Reaction12_thetaR': 12, 'Reaction13_alphaR': 13, 'Reaction14_alphaRp': 14, 'Reaction15_deltaMR': 15, 'Reaction16_betaR': 16}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.Reaction1(y, w, c, t), self.Reaction2(y, w, c, t), self.Reaction3(y, w, c, t), self.Reaction4(y, w, c, t), self.Reaction5(y, w, c, t), self.Reaction6(y, w, c, t), self.Reaction7(y, w, c, t), self.Reaction8(y, w, c, t), self.Reaction9(y, w, c, t), self.Reaction10(y, w, c, t), self.Reaction11(y, w, c, t), self.Reaction12(y, w, c, t), self.Reaction13(y, w, c, t), self.Reaction14(y, w, c, t), self.Reaction15(y, w, c, t), self.Reaction16(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def Reaction1(self, y, w, c, t):
		return y[0] * y[1] * c[1]


	def Reaction2(self, y, w, c, t):
		return y[0] * c[2]


	def Reaction3(self, y, w, c, t):
		return y[2] * c[3]


	def Reaction4(self, y, w, c, t):
		return y[1] * c[4]


	def Reaction5(self, y, w, c, t):
		return y[0] * y[4] * c[5]


	def Reaction6(self, y, w, c, t):
		return y[5] * c[6]


	def Reaction7(self, y, w, c, t):
		return y[4] * c[7]


	def Reaction8(self, y, w, c, t):
		return y[5] * c[8]


	def Reaction9(self, y, w, c, t):
		return y[6] * c[9]


	def Reaction10(self, y, w, c, t):
		return y[6] * c[10]


	def Reaction11(self, y, w, c, t):
		return y[0] * y[7] * c[11]


	def Reaction12(self, y, w, c, t):
		return y[8] * c[12]


	def Reaction13(self, y, w, c, t):
		return y[7] * c[13]


	def Reaction14(self, y, w, c, t):
		return y[8] * c[14]


	def Reaction15(self, y, w, c, t):
		return y[9] * c[15]


	def Reaction16(self, y, w, c, t):
		return y[9] * c[16]

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

	def __init__(self, y_indexes={'A': 0, 'R': 1, 'C': 2, 'EmptySet': 3, 'DA': 4, 'DAp': 5, 'MA': 6, 'DR': 7, 'DRp': 8, 'MR': 9}, w_indexes={}, c_indexes={'deterministicOscillator': 0, 'Reaction1_gammaC': 1, 'Reaction2_deltaA': 2, 'Reaction3_deltaA': 3, 'Reaction4_deltaR': 4, 'Reaction5_gammaA': 5, 'Reaction6_thetaA': 6, 'Reaction7_alphaA': 7, 'Reaction8_alphaAp': 8, 'Reaction9_deltaMA': 9, 'Reaction10_betaA': 10, 'Reaction11_gammaR': 11, 'Reaction12_thetaR': 12, 'Reaction13_alphaR': 13, 'Reaction14_alphaRp': 14, 'Reaction15_deltaMR': 15, 'Reaction16_betaR': 16}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 2.0, 1.0, 1.0, 0.2, 1.0, 50.0, 50.0, 500.0, 10.0, 50.0, 1.0, 100.0, 0.01, 50.0, 0.5, 5.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

