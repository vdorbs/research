from numpy import array, cos, linspace, pi, sin

from .ball import Ball
from .visual import Visual

class TwoBall(Ball, Visual):
    def __init__(self):
        Ball.__init__(self, 2)

    def boundary(self, N):
        angles = linspace(0, 2 * pi, N)
        return array([cos(angles), sin(angles)]).T
