from numpy import concatenate

from .gaussian_process_estimator import GaussianProcessEstimator
from ..util import arr_map

class ActionValueEstimator(GaussianProcessEstimator):
    def __init__(self, kernel, gamma, data, rewards, next_values):
        GaussianProcessEstimator.__init__(self, kernel, data)

        self.fit((1 - gamma) * rewards + gamma * next_values)

    def build(kernel, f, R, gamma, V_pi, states, actions):
        data = concatenate([states, actions], axis=1)
        next_states = arr_map(f, states, actions)
        rewards = arr_map(R, states, actions, next_states)
        next_values = arr_map(V_pi, next_states)

        return ActionValueEstimator(kernel, gamma, data, rewards, next_values)
