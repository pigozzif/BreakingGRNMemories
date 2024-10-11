import gymnasium
import numpy as np
import scipy.integrate
from gymnasium.spaces import Box


class MotionEquation(gymnasium.Env):

    def __init__(self, dim=2, dt=0.01, target=10, tol=1e-5):
        self.dim = dim
        self.dt = dt
        self.t = 0
        self.x = np.zeros(dim)
        self.target = np.full(self.x.shape, fill_value=target)
        self.tol = tol
        self.observation_space = Box(low=float("-inf"), high=float("inf"), shape=(self.dim,))
        self.action_space = Box(low=-1e2, high=1e2, shape=(self.dim,))

    def step(self, action):
        self.x = scipy.integrate.odeint(func=lambda y_t, t: action,
                                        y0=self.x,
                                        t=[self.t + i / 100 for i in range(int(1 / self.dt))])[-1, :]
        self.t += 1
        reward = -np.linalg.norm(self.x - self.target)
        done = -reward < self.tol or self.t > 1000
        return self.x, reward, done, False, {}

    def render(self):
        return None

    def reset(self, *, seed=None, options=None):
        self.x = np.zeros(self.dim)
        return self.x, {}

    def close(self):
        pass
