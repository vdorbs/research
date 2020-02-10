from numpy import dot

from .kernel import Kernel

class PolynomialKernel(Kernel):
    def __init__(self, c=0, q=1):
        self.c, self.q = c, q

    def K(self, x, y):
        return (self.c + dot(x, y)) ** self.q

    def label(self):
        return 'Polynomial Kernel $(c = ' + str(self.c) + ', q = ' + str(self.q) + ')$'
