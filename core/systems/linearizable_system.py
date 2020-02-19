from .ivp_system import IVPSystem

class LinearizableSystem(IVPSystem):
    def __init__(self, dt):
        IVPSystem.__init__(self, dt)

    def dFdx(self, x, a):
        raise NotImplementedError

    def dFda(self, x, a):
        raise NotImplementedError
