import numpy as np
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        self._v = None

    def update(self,params,grads):
        if self._v is None:
            self._v = {}
            for key,val in params.items():
                self._v[key] = np.zeros_like(val)
        for key in params.keys():
            self._v[key] = self._momentum * (self._v[key]) - self._lr * grads[key]
            params[key] += self._v[key]
