from numpy import array

class Kernel:
    def K(self, x, y):
        raise NotImplementedError

    def label(self):
        raise NotImplementedError

    def matrix(self, xs):
         return array([[self.K(x_i, x_j) for x_i in xs] for x_j in xs])
