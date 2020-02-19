from numpy import array, cos, linspace, meshgrid, pi, reshape, sin

from .ball import Ball
from .visual import Visual

class ThreeBall(Ball, Visual):
    def __init__(self):
        Ball.__init__(self, 3)

    def boundary_grid(self, N):
        lats = linspace(-pi / 2, pi / 2, N)
        longs = linspace(-pi, pi, N)
        lats, longs = meshgrid(lats, longs)

        zs = sin(lats)
        rs = cos(lats)
        xs = rs * cos(longs)
        ys = rs * sin(longs)
        return xs, ys, zs

    def boundary(self, N):
        xs, ys, zs = self.boundary_grid(N)
        return array(list(map(lambda arr: reshape(arr, -1), [xs, ys, zs]))).T

    def plot_boundary(self, ax, N):
        return ax.plot_surface(*self.boundary_grid(N), alpha=0.05)
