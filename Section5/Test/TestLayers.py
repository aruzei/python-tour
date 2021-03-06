import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Layers.Add import *
from Layers.Mutiple import *
from Layers.Sigmoid import *
from Layers.Affine import *
from Layers.SoftMaxWithLoss import *

# z = sigma((w * x) + b ) 
w = 1
x = 1
b = 1

multiple = Mutiple()
add = Add()
sigmoid = Sigmoid()

z = multiple.forward(w, x)
print("z =" ,z)
h = add.forward(z,b)
print("h =" ,h)
y = sigmoid.forward(h)
print("y =" ,y)

dy_h = sigmoid.backward(1)
print("dy/dh =" ,dy_h)
dy_z = add.backward(dy_h)
print("dy/db =",dy_z[1])
print("dy/dwx =" ,dy_z[0])

dy_w,dy_x = multiple.backward(dy_z[0])
print("dy/dw =" ,dy_w)
print("dy/dx =" ,dy_x)

affine = Affine(np.array([1]),np.array([1]))
ha = affine.forward(np.array([1]))
print ("ha == h:",ha==h)

h_loss = SoftMaxWithLoss()
L = h_loss.forward(ha,np.array([1]))
print(L)

print(h_loss.backward())