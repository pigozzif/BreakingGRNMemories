import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y_indexes = {'P0': 0, 'P1': 1, 'T0': 2, 'T1': 3, 'P2': 4, 'T2': 5, 'CC': 6, 'Cn': 7, 'Mp': 8, 'Mt': 9}

w0 = jnp.array([0.0, 0.0])
w_indexes = {'Pt': 0, 'Tt': 1}

c = jnp.array([0.7, 2.0, 1.0, 1.0, 2.0, 8.0, 2.0, 8.0, 2.0, 1.0, 2.0, 1.0, 2.0, 8.0, 2.0, 8.0, 2.0, 1.0, 2.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 2.0, 0.2, 0.01, 0.2, 1.2, 0.6, 0.6, 0.2, 0.01, 0.01, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 0.9, 0.9, 0.01, 0.7, 0.2, 0.01, 0.2]) 
c_indexes = {'V_mT': 0, 'V_dT': 1, 'Cell': 2, 'compartment_0000002': 3, 'P0_to_P1_K1_P': 4, 'P0_to_P1_V_1P': 5, 'T0_to_T1_K_1T': 6, 'T0_to_T1_V_1T': 7, 'P1_to_P0_K_2P': 8, 'P1_to_P0_V_2P': 9, 'T1_to_T0_K_2T': 10, 'T1_to_T0_V_2T': 11, 'P1_to_P2_K_3P': 12, 'P1_to_P2_V_3P': 13, 'T1_to_T2_K_3T': 14, 'T1_to_T2_V_3T': 15, 'P2_to_P1_K_4P': 16, 'P2_to_P1_V_4P': 17, 'T2_to_T1_K_4T': 18, 'T2_to_T1_V_4T': 19, 'P0_degradation_k_d': 20, 'T0_degradation_k_d': 21, 'P1_degradation_k_d': 22, 'T1_degradation_k_d': 23, 'P2_degradation_k_d': 24, 'P2_degradation_V_dP': 25, 'P2_degradation_K_dP': 26, 'T2_degradation_k_d': 27, 'T2_degradation_K_dT': 28, 'PT_complex_formation_k3': 29, 'PT_complex_formation_k4': 30, 'PT_complex_nucleation_k1': 31, 'PT_complex_nucleation_k2': 32, 'PT_complex_degradation_k_dC': 33, 'PTnucl_complex_degradation_k_dN': 34, 'Mp_production_v_sP': 35, 'Mp_production_K_IP': 36, 'Mp_production_n': 37, 'Mt_production_V_sT': 38, 'Mt_production_K_IT': 39, 'Mt_production_n': 40, 'P0_production_k_sP': 41, 'T0_production_k_sT': 42, 'Mp_degradation_k_d': 43, 'Mp_degradation_V_mP': 44, 'Mp_degradation_K_mP': 45, 'Mt_degradation_k_d': 46, 'Mt_degradation_K_mT': 47}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.P0_to_P1(y, w, c, t), self.T0_to_T1(y, w, c, t), self.P1_to_P0(y, w, c, t), self.T1_to_T0(y, w, c, t), self.P1_to_P2(y, w, c, t), self.T1_to_T2(y, w, c, t), self.P2_to_P1(y, w, c, t), self.T2_to_T1(y, w, c, t), self.P0_degradation(y, w, c, t), self.T0_degradation(y, w, c, t), self.P1_degradation(y, w, c, t), self.T1_degradation(y, w, c, t), self.P2_degradation(y, w, c, t), self.T2_degradation(y, w, c, t), self.PT_complex_formation(y, w, c, t), self.PT_complex_nucleation(y, w, c, t), self.PT_complex_degradation(y, w, c, t), self.PTnucl_complex_degradation(y, w, c, t), self.Mp_production(y, w, c, t), self.Mt_production(y, w, c, t), self.P0_production(y, w, c, t), self.T0_production(y, w, c, t), self.Mp_degradation(y, w, c, t), self.Mt_degradation(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def P0_to_P1(self, y, w, c, t):
		return c[2] * c[5] * (y[0]/1.0) / (c[4] + (y[0]/1.0))


	def T0_to_T1(self, y, w, c, t):
		return c[2] * c[7] * (y[2]/1.0) / (c[6] + (y[2]/1.0))


	def P1_to_P0(self, y, w, c, t):
		return c[2] * c[9] * (y[1]/1.0) / (c[8] + (y[1]/1.0))


	def T1_to_T0(self, y, w, c, t):
		return c[2] * c[11] * (y[3]/1.0) / (c[10] + (y[3]/1.0))


	def P1_to_P2(self, y, w, c, t):
		return c[2] * c[13] * (y[1]/1.0) / (c[12] + (y[1]/1.0))


	def T1_to_T2(self, y, w, c, t):
		return c[2] * c[15] * (y[3]/1.0) / (c[14] + (y[3]/1.0))


	def P2_to_P1(self, y, w, c, t):
		return c[2] * c[17] * (y[4]/1.0) / (c[16] + (y[4]/1.0))


	def T2_to_T1(self, y, w, c, t):
		return c[2] * c[19] * (y[5]/1.0) / (c[18] + (y[5]/1.0))


	def P0_degradation(self, y, w, c, t):
		return c[2] * c[20] * (y[0]/1.0)


	def T0_degradation(self, y, w, c, t):
		return c[2] * c[21] * (y[2]/1.0)


	def P1_degradation(self, y, w, c, t):
		return c[2] * c[22] * (y[1]/1.0)


	def T1_degradation(self, y, w, c, t):
		return c[2] * c[23] * (y[3]/1.0)


	def P2_degradation(self, y, w, c, t):
		return c[2] * c[24] * (y[4]/1.0) + c[2] * c[25] * (y[4]/1.0) / (c[26] + (y[4]/1.0))


	def T2_degradation(self, y, w, c, t):
		return c[2] * c[27] * (y[5]/1.0) + c[2] * c[1] * (y[5]/1.0) / (c[28] + (y[5]/1.0))


	def PT_complex_formation(self, y, w, c, t):
		return c[2] * c[29] * (y[4]/1.0) * (y[5]/1.0) - c[2] * c[30] * (y[6]/1.0)


	def PT_complex_nucleation(self, y, w, c, t):
		return c[2] * c[31] * (y[6]/1.0) - c[3] * c[32] * (y[7]/1.0)


	def PT_complex_degradation(self, y, w, c, t):
		return c[2] * c[33] * (y[6]/1.0)


	def PTnucl_complex_degradation(self, y, w, c, t):
		return c[3] * c[34] * (y[7]/1.0)


	def Mp_production(self, y, w, c, t):
		return c[2] * c[35] * c[36]**c[37] / (c[36]**c[37] + (y[7]/1.0)**c[37])


	def Mt_production(self, y, w, c, t):
		return c[2] * c[38] * c[39]**c[40] / (c[39]**c[40] + (y[7]/1.0)**c[40])


	def P0_production(self, y, w, c, t):
		return c[2] * c[41] * (y[8]/1.0)


	def T0_production(self, y, w, c, t):
		return c[2] * c[42] * (y[9]/1.0)


	def Mp_degradation(self, y, w, c, t):
		return c[2] * c[43] * (y[8]/1.0) + c[2] * c[44] * (y[8]/1.0) / (c[45] + (y[8]/1.0))


	def Mt_degradation(self, y, w, c, t):
		return c[2] * c[46] * (y[9]/1.0) + c[2] * c[0] * (y[9]/1.0) / (c[47] + (y[9]/1.0))

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((y[6]/1.0) + (y[7]/1.0) + (y[0]/1.0) + (y[1]/1.0) + (y[4]/1.0)))

		w = w.at[1].set(((y[6]/1.0) + (y[7]/1.0) + (y[2]/1.0) + (y[3]/1.0) + (y[5]/1.0)))

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

	def __init__(self, y_indexes={'P0': 0, 'P1': 1, 'T0': 2, 'T1': 3, 'P2': 4, 'T2': 5, 'CC': 6, 'Cn': 7, 'Mp': 8, 'Mt': 9}, w_indexes={'Pt': 0, 'Tt': 1}, c_indexes={'V_mT': 0, 'V_dT': 1, 'Cell': 2, 'compartment_0000002': 3, 'P0_to_P1_K1_P': 4, 'P0_to_P1_V_1P': 5, 'T0_to_T1_K_1T': 6, 'T0_to_T1_V_1T': 7, 'P1_to_P0_K_2P': 8, 'P1_to_P0_V_2P': 9, 'T1_to_T0_K_2T': 10, 'T1_to_T0_V_2T': 11, 'P1_to_P2_K_3P': 12, 'P1_to_P2_V_3P': 13, 'T1_to_T2_K_3T': 14, 'T1_to_T2_V_3T': 15, 'P2_to_P1_K_4P': 16, 'P2_to_P1_V_4P': 17, 'T2_to_T1_K_4T': 18, 'T2_to_T1_V_4T': 19, 'P0_degradation_k_d': 20, 'T0_degradation_k_d': 21, 'P1_degradation_k_d': 22, 'T1_degradation_k_d': 23, 'P2_degradation_k_d': 24, 'P2_degradation_V_dP': 25, 'P2_degradation_K_dP': 26, 'T2_degradation_k_d': 27, 'T2_degradation_K_dT': 28, 'PT_complex_formation_k3': 29, 'PT_complex_formation_k4': 30, 'PT_complex_nucleation_k1': 31, 'PT_complex_nucleation_k2': 32, 'PT_complex_degradation_k_dC': 33, 'PTnucl_complex_degradation_k_dN': 34, 'Mp_production_v_sP': 35, 'Mp_production_K_IP': 36, 'Mp_production_n': 37, 'Mt_production_V_sT': 38, 'Mt_production_K_IT': 39, 'Mt_production_n': 40, 'P0_production_k_sP': 41, 'T0_production_k_sT': 42, 'Mp_degradation_k_d': 43, 'Mp_degradation_V_mP': 44, 'Mp_degradation_K_mP': 45, 'Mt_degradation_k_d': 46, 'Mt_degradation_K_mT': 47}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), w0=jnp.array([0.0, 0.0]), c=jnp.array([0.7, 2.0, 1.0, 1.0, 2.0, 8.0, 2.0, 8.0, 2.0, 1.0, 2.0, 1.0, 2.0, 8.0, 2.0, 8.0, 2.0, 1.0, 2.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 2.0, 0.2, 0.01, 0.2, 1.2, 0.6, 0.6, 0.2, 0.01, 0.01, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 0.9, 0.9, 0.01, 0.7, 0.2, 0.01, 0.2]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

