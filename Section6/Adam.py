import numpy as np
class Adam:
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.999):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._momentum = None
        self._v = None
        self._iteration = None
        
    def update(self,params,grads):
        if self._momentum is None:
            self._momentum = {}
            for key,val in params.items():
                self._momentum[key] = np.zeros_like(val)
        if self._v is None:
            self._v = {}
            for key,val in params.items():
                self._v[key] = np.zeros_like(val)
        if self._iteration is None:
            self._iteration = 1
        for key in params.keys():
            self._momentum[key] = self._beta1 * self._momentum[key] + (1-self._beta1) * grads[key]
            self._v[key] = self._beta2 * self._v[key] + (1-self._beta2) * grads[key] * grads[key]
            
            moumentum_corrected = self._momentum[key] / (1 - np.power(self._beta1,self._iteration)) 
            velocity_corrected = self._v[key] / (1 - np.power(self._beta2,self._iteration)) 

            params[key] -= self._lr * moumentum_corrected / (np.sqrt(velocity_corrected) + 10e-7)
