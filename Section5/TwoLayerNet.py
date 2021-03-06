import numpy as np
from Layers.Affine import Affine
from Layers.ReLU import ReLU

from Layers.SoftMaxWithLoss import SoftMaxWithLoss

from collections import OrderedDict

class TwoLayerNet: 
    def __init__(self, input_size, hidden_size, output_size,
                weight_init_std = 0.01):
        self._params = {}
        self._params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self._params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self._params["b1"] = np.zeros(hidden_size)
        self._params["b2"] = np.zeros(output_size)

        self._layers = OrderedDict()
        self._layers['Affine1'] = Affine(self._params["W1"],self._params["b1"])
        self._layers["ReLU1"] = ReLU()
        self._layers['Affine2'] = Affine(self._params["W2"],self._params["b2"])
        self._lastLayer = SoftMaxWithLoss()

    def predict(self,x):
        for layer in self._layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self._lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t,axis=1)

        acc = np.sum(y == t) / float(x.shape[0])

        return acc
    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)
            
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self._params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self._params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self._params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self._params['b2'])
        
        return grads
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self._lastLayer.backward(dout)

        layers = list(self._layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self._layers["Affine1"].dW
        grads['W2'] = self._layers["Affine2"].dW
        grads['b1'] = self._layers["Affine1"].db
        grads['b2'] = self._layers["Affine2"].db

        return grads
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad