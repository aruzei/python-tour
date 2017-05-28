# coding: utf-8
import numpy as np

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mnist import load_mnist
from MultiLayerNet import MutiLayerNet

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
        
        if i % (10) == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    return train_loss_list,train_acc_list,test_acc_list
def Prepare_training(x_train,t_train,x_test,t_test):
    
    def train_NN(optimizer,hidden_size_list,iters_num,weight_init_std):
        network = MutiLayerNet(input_size=x_train.shape[1], hidden_size_list= hidden_size_list, output_size=t_train.shape[1],weight_init_std=weight_init_std)
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

train_loss_list1,train_acc_list1,test_acc_list1 = training(optimizer1,[50,50,50],200,0.01)
train_loss_list2,train_acc_list2,test_acc_list2 = training(optimizer2,[50,50,50],200,0.01)
train_loss_list3,train_acc_list3,test_acc_list3 = training(optimizer4,[50,50,50],200,0.01)

ones = np.ones(5)/5.0

plt.subplot(1,2,1)
plt.plot(np.convolve(train_loss_list1,ones,'valid'),label="SGD")
plt.plot(np.convolve(train_loss_list2,ones,'valid'),label="Momentum")
plt.plot(np.convolve(train_loss_list3,ones,'valid'),label="Adam")

plt.subplot(1,2,2)

plt.plot(train_acc_list1,label="SGD")
plt.plot(train_acc_list2,label="Momentum")
plt.plot(train_acc_list3,label="Adam")

plt.legend()
plt.show()