from numpy import array, pi, sin

from .affine_system import AffineSystem

class Pendulum(AffineSystem):
    def __init__(self, m, l, dt, g=9.81):
        self.m, self.l, self.dt, self.g = m, l, dt, g

    def F_0(self, x):
        theta, theta_dot = x
        return array([theta_dot, -self.g * sin(theta) / self.l])

    def G(self, x):
        return array([[0], [1 / (self.m * self.l ** 2)]])
