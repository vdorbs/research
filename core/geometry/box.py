from numpy import concatenate, identity, Inf, linspace, meshgrid, ones
from numpy.linalg import norm
from numpy.ma import masked_array
from numpy.random import rand

from .primitive import Primitive

class Box(Primitive):
    def __init__(self, d):
        Primitive.__init__(self, d)

    def sample(self, N=1):
        return 2 * rand(N, self.d) - 1

    def meshgrid(self, N):
        grid_1d = linspace(-1, 1, N)
        return meshgrid(*([grid_1d] * self.d), indexing='ij')

    def mask(self, N):
        mask = False * ones([N] * self.d)
        return tuple(map(lambda grid: masked_array(grid, mask), self.meshgrid(N)))

    def label(self):
        return '$' + str(self.d) + '$-box'

    def is_member(self, xs):
        return norm(xs, Inf, axis=1) <= 1

    def volume(self):
        return 2 ** self.d

    def barrier(self, x):
        return concatenate([1 + x, 1 - x])

    def barrier_jac(self, x):
        return concatenate([identity(len(x)), -identity(len(x))])
