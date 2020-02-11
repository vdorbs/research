from numpy import dot
from numpy.linalg import det, inv

from .convex_body import ConvexBody

class Derived(ConvexBody):
    def __init__(self, primitive, T, c):
        self.primitive = primitive
        self.T = T
        self.T_inv = inv(T)
        self.c = c

    def to_primitive(self, xs):
        return dot(self.T, (xs - self.c).T).T

    def from_primitive(self, xs):
        return dot(self.T_inv, xs.T).T + self.c

    def sample(self, N):
        return self.from_primitive(self.primitive.sample(N))

    def label(self):
        return 'Derived from ' + self.primitive.label()

    def volume(self):
        return abs(det(self.T_inv)) * self.primitive.volume()

    def barrier(self, x):
        return self.primitive.barrier(self.to_primitive(x))

    def barrier_grad(self, x):
        return dot(self.T.T, self.primitive.barrier_grad(self.to_primitive(x)))
