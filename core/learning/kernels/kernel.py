from numpy import array

from ...util import arr_map

class Kernel:
    def K(self, x, y):
        raise NotImplementedError

    def label(self):
        raise NotImplementedError

    def embedding(self, xs):

        def phi(self, x):
            return array([self.K(x_i, x) for x_i in xs])

        return phi

    def matrix(self, xs):
        phi = self.embedding(xs)
        return arr_map(phi, xs)
