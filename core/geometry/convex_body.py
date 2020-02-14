from numpy import concatenate, mean
from numpy.linalg import norm

from .body import Body

class ConvexBody(Body):
    def __init__(self, d):
        self.d = d

    def distances(self, xs, x):
        return norm(xs - x, axis=1)

    def centroid(self, xs):
        return mean(xs, axis=0)

    def is_member(self, xs):
        raise NotImplementedError

    def volume(self):
        raise NotImplementedError
