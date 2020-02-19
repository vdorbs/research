from numpy import concatenate, mean, zeros
from numpy.linalg import norm
from numpy.ma import masked_array

from .body import Body
from ..util import arr_map

class ConvexBody(Body):
    def __init__(self, d):
        self.d = d

    def distances(self, xs, x):
        return norm(xs - x, axis=1)

    def centroid(self, xs):
        return mean(xs, axis=0)

    def is_member(self, xs):
        raise NotImplementedError

    def mask(self, N):
        raise NotImplementedError

    def volume(self):
        raise NotImplementedError

    def compressed(self, masked_grids):
        return arr_map(lambda masked_grid: masked_grid.compressed(), masked_grids).T

    def masked_apply(self, func, masked_grids):
        xs = self.compressed(masked_grids)
        zs = arr_map(func, xs)

        masked_grid = masked_grids[0]
        res = masked_array(zeros(masked_grid.shape), mask=masked_grid.mask)
        res[~res.mask] = zs
        return res
