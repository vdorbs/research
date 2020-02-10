from numpy import array, min, pi
from numpy.random import rand

from .body import Body

class OneSphere(Body):
    def sample(self, N):
        return pi * (2 * rand(N) - 1)

    def distances(self, xs, x):
        dists = abs(xs - x)
        dists = min(array([dists, 2 * pi - dists]), axis=0)
        return dists

    def centroid(self, xs):
        # TODO: Implement

        raise NotImplementedError

    def label(self):
        return '$\\mathbb{S}^1$'
