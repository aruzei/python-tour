import numpy as np
class AdaGrad:
    def __init__(self,lr=0.01):
        self._lr = lr
        self._h = None

    def update(self,params,grads):
        if self._h is None:
            self._h = {}
            for key,val in params.items():
                self._h[key] = np.zeros_like(val)
        for key in params.keys():
            self._h[key] += grads[key]*grads[key]
            params[key] -= self._lr * grads[key] / (np.sqrt(self._h[key])+1e-7)
