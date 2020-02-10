from numpy import identity, zeros
from numpy.linalg import norm
from numpy.random import multivariate_normal, rand

from .primitive import Primitive

class Ball(Primitive):
    def __init__(self, d):
        self.d = d

    def sample(self, N=1):
        gaussian_samples = multivariate_normal(zeros(self.d), identity(self.d), N)
        sphere_samples = gaussian_samples / norm(gaussian_samples, axis=1, keepdims=True)
        samples = (rand(N, 1) ** (1 / self.d)) * sphere_samples
        return samples

    def label(self):
         return '$' +  str(self.d) + '$-ball'

    def barrier(self, x):
        return 1 - norm(x) ** 2

    def barrier_grad(self, x):
        return -2 * x
