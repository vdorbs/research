from numpy import dot

from .ivp_system import IVPSystem

class AffineSystem(IVPSystem):
    def __init__(self, dt):
        IVPSystem.__init__(self, dt)

    def F_0(self, x):
        raise NotImplementedError

    def G(self, x):
        raise NotImplementedError

    def F(self, x, a):
        return self.F_0(x) + dot(self.G(x), a)
