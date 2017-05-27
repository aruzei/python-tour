
class Mutiple:
    def __init__(self):
        self._x = None
        self._y = None
        
    def forward(self, x, y):
        self._x = x
        self._y = y
        return x * y
        
    def backward(self, dz):
        dx = dz * self._y
        dy = dz * self._x
        return dx,dy