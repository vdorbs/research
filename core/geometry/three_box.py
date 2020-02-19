from numpy import array, concatenate, ones

from .box import Box
from .two_box import TwoBox
from .visual import Visual

class ThreeBox(Box, Visual):
    def __init__(self):
        Box.__init__(self, 3)

    def boundary(self, N=None):
        face = TwoBox().boundary()
        one = ones((len(face), 1))
        x_neg_face = concatenate([-one, face], axis=1)
        x_pos_face = concatenate([one, face], axis=1)
        x_to_y = array([[1, 1, -1]])
        y_pos_face = concatenate([face[:, :1], one, face[:, 1:]], axis=1)
        y_neg_face = concatenate([face[:, :1], -one, face[:, 1:]], axis=1)
        return concatenate([x_neg_face, x_pos_face, x_to_y, y_pos_face, y_neg_face])
