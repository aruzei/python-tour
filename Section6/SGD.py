
class SGD:
    def __init__(self,lr=0.01):
        self._lr = lr
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self._lr * grads[key]