from numpy import mean
from numpy.linalg import norm

from .body import Body

class ConvexBody(Body):
    def distances(self, xs, x):
        return norm(xs - x, axis=1)

    def centroid(self, xs):
        return mean(xs, axis=0)

    def volume(self):
        raise NotImplementedError
