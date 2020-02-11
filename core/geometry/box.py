from numpy import concatenate, identity
from numpy.random import rand

from .primitive import Primitive

class Box(Primitive):
    def __init__(self, d):
        self.d = d

    def label(self):
        return '$' + str(self.d) + '$-box'

    def sample(self, N=1):
        return 2 * rand(N, self.d) - 1

    def volume(self):
        return 2 ** self.d

    def barrier(self, x):
        # return concatenate([1 + x, 1 - x])

        raise NotImplementedError

    def barrier_grad(self, x):
        # return concatenate([identity(len(x)), -identity(len(x))])

        raise NotImplementedError
