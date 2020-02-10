from numpy import dot

from .policy import Policy

class LinearPolicy(Policy):
    def __init__(self, K):
        self.K = K

    def pi(self, s):
        return -dot(self.K, s)
