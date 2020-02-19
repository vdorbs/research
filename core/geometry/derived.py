from numpy import dot, zeros
from numpy.linalg import det, inv

from .convex_body import ConvexBody

class Derived(ConvexBody):
    def __init__(self, primitive, T, c):
        ConvexBody.__init__(self, primitive.d)
        self.primitive = primitive
        self.T = T
        self.T_inv = inv(T)
        self.c = c

    def to_primitive(self, xs):
        return dot(self.T, (xs - self.c).T).T

    def from_primitive(self, xs):
        return dot(self.T_inv, xs.T).T + self.c

    def is_member(self, xs):
        return self.primitive.is_member(self.to_primitive(xs))

    def sample(self, N):
        return self.from_primitive(self.primitive.sample(N))

    def mask(self, N):
        masked_grids = self.primitive.mask(N)
        zs = self.from_primitive(self.compressed(masked_grids))
        for i, (masked_grid, xs) in enumerate(zip(masked_grids, zs.T)):
            masked_grids[i][~masked_grid.mask] = xs

        return masked_grids

    def label(self):
        return 'Derived from ' + self.primitive.label()

    def volume(self):
        return abs(det(self.T_inv)) * self.primitive.volume()

    def barrier(self, x):
        return self.primitive.barrier(self.to_primitive(x))

    def barrier_grad(self, x):
        return dot(self.T.T, self.primitive.barrier_grad(self.to_primitive(x)))
