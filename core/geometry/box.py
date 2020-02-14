from numpy import concatenate, identity, Inf
from numpy.linalg import norm
from numpy.random import rand

from .primitive import Primitive

class Box(Primitive):
    def __init__(self, d):
        Primitive.__init__(self, d)

    def sample(self, N=1):
        return 2 * rand(N, self.d) - 1

    def label(self):
        return '$' + str(self.d) + '$-box'

    def is_member(self, xs):
        return norm(xs, Inf, axis=1) <= 1

    def volume(self):
        return 2 ** self.d

    def barrier(self, x):
        # return concatenate([1 + x, 1 - x])

        raise NotImplementedError

    def barrier_grad(self, x):
        # return concatenate([identity(len(x)), -identity(len(x))])

        raise NotImplementedError
