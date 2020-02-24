from numpy import reshape

from .three_ball import ThreeBall
from .visual_derived import VisualDerived

class ThreeEllipsoid(VisualDerived):
    def __init__(self, T, c):
        VisualDerived.__init__(self, ThreeBall(), T, c)

    def label(self):
        return '$3$-ellipsoid'

    def plot_boundary(self, ax, N, color=None, linewidth=None, label=''):
        xs, ys, zs = map(lambda arr: reshape(arr, (N, N)), self.boundary(N).T)
        return ax.plot_surface(xs, ys, zs, alpha=0.05, label=label)
