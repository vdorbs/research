from numpy import dot
from numpy.linalg import solve

from ..util import arr_map

class GaussianProcessEstimator:
    def __init__(self, kernel, data):
        self.embedding = kernel.embedding(data)
        self.kernel_mat = arr_map(self.embedding, data)
        self.weights = None

    def fit(self, targets):
        self.weights = solve(self.kernel_mat, targets)
        return self

    def eval(self, x):
        return dot(self.weights, self.embedding(x))
