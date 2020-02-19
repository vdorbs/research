from numpy import array, cos, pi, sin

from .affine_system import AffineSystem
from .linearizable_system import LinearizableSystem

class Pendulum(AffineSystem, LinearizableSystem):
    def __init__(self, m, l, dt, g=9.81):
        AffineSystem.__init__(self, dt)
        LinearizableSystem.__init__(self, dt)
        self.m, self.l, self.g = m, l, g

    def F_0(self, x):
        theta, theta_dot = x
        return array([theta_dot, -self.g * sin(theta) / self.l])

    def G(self, x):
        return array([[0], [1 / (self.m * self.l ** 2)]])

    def dFdx(self, x, a):
        theta, _ = x
        return array([[0, 1], [-self.g * cos(theta) / self.l, 0]])

    def dFda(self, x, a):
        return self.G(x)
