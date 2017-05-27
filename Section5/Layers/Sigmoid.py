import numpy as np
class Sigmoid:
    def forward(self, x):

        self._out = 1 / (1 + np.exp(-x))

        return self._out
        
    def backward(self, dz):
        dx = dz * (1 - self._out) * self._out
        return dx