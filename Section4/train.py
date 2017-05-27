import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

import numpy as np
from TwoLayerNet import TwoLayerNet
x = np.random.rand(100,784)
t = np.random.rand(100,10)

(x_train,t_train) ,(x_test,t_test) = load_mnist(flatten= True,normalize = True)

train_loss_list =[]
iters_num = 1
batch_size = 100
train_size = 10
learning_rate = 0.1

network = TwoLayerNet(input_size = 784,hidden_size = 50,output_size = 10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    
    train_loss_list.append(loss)
    print(loss)