from .convex_body import ConvexBody

class Primitive(ConvexBody):
    def barrier(self, x):
        raise NotImplementedError

    def barrier_grad(self, x):
        raise NotImplementedError
