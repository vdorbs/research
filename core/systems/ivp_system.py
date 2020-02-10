from scipy.integrate import solve_ivp

from .system import System

class IVPSystem(System):
    def __init__(self, dt):
        self.dt = dt

    def F(self, x, a):
        raise NotImplementedError

    def step(self, s, a):

        def dx(t, x):
            return self.F(x, a)

        solution = solve_ivp(dx, [0, self.dt], s).y
        return solution[:, -1]
