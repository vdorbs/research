from numpy import identity, pi, zeros
from numpy.linalg import norm
from numpy.random import multivariate_normal, rand

from .primitive import Primitive

class Ball(Primitive):
    def __init__(self, d):
        Primitive.__init__(self, d)

    def sample(self, N=1):
        gaussian_samples = multivariate_normal(zeros(self.d), identity(self.d), N)
        sphere_samples = gaussian_samples / norm(gaussian_samples, axis=1, keepdims=True)
        samples = (rand(N, 1) ** (1 / self.d)) * sphere_samples
        return samples

    def label(self):
         return '$' +  str(self.d) + '$-ball'

    def is_member(self, xs):
        return norm(xs, axis=1) <= 1

    def volume(self):
        if self.d is 1:
            return 2

        if self.d is 2:
            return pi

        return 2 * pi * Ball(self.d - 2).volume() / self.d

    def barrier(self, x):
        return 1 - norm(x) ** 2

    def barrier_grad(self, x):
        return -2 * x
