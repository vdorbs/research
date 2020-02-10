from cvxpy import Minimize, Problem, sum_squares, Variable
from numpy import dot
from numpy.linalg import norm

from .policy import Policy

class SafePolicy(Policy):
    def __init__(self, primitive, system, m, policy, alpha=1):
        self.primitive = primitive
        self.system = system
        self.policy = policy
        self.alpha = alpha
        self.a = Variable(m)

    def pi(self, s):
        obj = Minimize(sum_squares(self.a - self.policy.pi(s)))
        barrier_derivative = self.primitive.barrier_grad(s) * (self.system.F_0(s) + self.system.G(s) * self.a)
        cons = [ barrier_derivative >= -self.alpha * self.primitive.barrier(s) ]
        prob = Problem(obj, cons)
        prob.solve()
        return self.a.value
