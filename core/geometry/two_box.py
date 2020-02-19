from numpy import array

from .box import Box
from .visual import Visual

class TwoBox(Box, Visual):
    def __init__(self):
        Box.__init__(self, 2)

    def boundary(self, N=None):
        xs = array([-1, 1, 1, -1, -1])
        ys = array([-1, -1, 1, 1, -1])
        return array([xs, ys]).T
