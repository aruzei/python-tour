import numpy as np

class Affine:
    def __init__(self, W,b):
        self._W = W
        self._b = b
        self._X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self._X = X
        
        out = np.dot(X,self._W) + self._b
        return out

    def backward(self, dH):
        dX = np.dot(dH,self._W.T)
        self.dW = np.dot(self._X.T,dH)
        self.db = np.sum(dH,axis=0)
        return dX
