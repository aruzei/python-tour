
class ReLU:
    def forward(self,x):
        self._mask = (x <= 0)
        out = x.copy()
        return out[self._mask]
        
    def backward(self,z):

        z[self._mask] = 0
        dx = z * 1
        return dx