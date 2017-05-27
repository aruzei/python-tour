# coding: utf-8
import numpy as np

class SoftMaxWithLoss:
    def __init__(self):
        self._loss = None
        self._y = None
        self._t = None
    
    def forward(self, x, t):
        self._t = t
        self._y = _softmax(x)
        self._loss = _cross_entropy_error(self._y,self._t)
        return self._loss
    def backward(self,dH = 1):
        batch_size = self._t.shape[0]
        dx = (self._y - self._t) / batch_size
        return dx
def _softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # avoide overflow
    return np.exp(x) / np.sum(np.exp(x))
def _cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y) / batch_size)
