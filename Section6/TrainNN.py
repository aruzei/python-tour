# coding: utf-8
import numpy as np

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mnist import load_mnist
from Section5.TwoLayerNet import TwoLayerNet

def train(x_train,t_train,x_test,t_test,network,optimizer,iters_num):
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 勾配
        #grad = network.numerical_gradient(x_batch, t_batch)
        grads = network.gradient(x_batch, t_batch)
        params = network._params
        # 更新
        optimizer.update(params,grads)
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    return train_loss_list
def Prepare_training(x_train,t_train,x_test,t_test):
    
    def train_NN(optimizer,iters_num):
        network = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=t_train.shape[1])
        return train(x_train,t_train,x_test,t_test,network,optimizer,iters_num)
    return train_NN

import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

training = Prepare_training(x_train,t_train,x_test,t_test)

from SGD import SGD
from Momentum import Momentum
from AdaGrad import AdaGrad
from Adam import Adam

optimizer1 = SGD(0.01)
optimizer2 = Momentum(0.01,0.9)
optimizer3 = AdaGrad(0.01)
optimizer4 = Adam(0.01)


train_loss_list1 = training(optimizer1,2000)
train_loss_list2 = training(optimizer2,2000)
train_loss_list3 = training(optimizer3,2000)
train_loss_list4 = training(optimizer4,2000)


plt.plot(train_loss_list1[::1],label="SGD")
plt.plot(train_loss_list2[::1],label="Momentum")
plt.plot(train_loss_list3[::1],label="AdaGrad")
plt.plot(train_loss_list4[::1],label="Adam")

plt.legend()
plt.show()