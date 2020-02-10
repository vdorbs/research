from numpy import exp
from numpy.linalg import norm

from .kernel import Kernel

class LaplacianKernel(Kernel):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def K(self, x, y):
        return exp(-self.alpha * norm(x - y, 1))

    def label(self):
        return 'Laplacian Kernel $(\\alpha = ' + str(self.alpha) + ')$'
