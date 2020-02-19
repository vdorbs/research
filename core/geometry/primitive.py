from .convex_body import ConvexBody
from ..util import arr_map

class Primitive(ConvexBody):
    def __init__(self, d):
        ConvexBody.__init__(self, d)

    def barrier(self, x):
        raise NotImplementedError

    def barrier_grad(self, x):
        raise NotImplementedError
