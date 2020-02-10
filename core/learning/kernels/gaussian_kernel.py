from numpy import exp
from numpy.linalg import norm

from .kernel import Kernel

class GaussianKernel(Kernel):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def K(self, x, y):
        return exp(-(norm(x - y) ** 2) / (2 * self.sigma ** 2))

    def label(self):
        return 'Gaussian Kernel $(\\sigma = ' + str(self.sigma) + ')$'
