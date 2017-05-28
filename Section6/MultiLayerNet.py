import numpy as np
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Section5.Layers.Affine import Affine
from Section5.Layers.ReLU import ReLU

from BatchNormalization import BatchNormalization

from Section5.Layers.SoftMaxWithLoss import SoftMaxWithLoss

from collections import OrderedDict

class MutiLayerNet: 
    def __init__(self, input_size, hidden_size_list, output_size,
                weight_init_std):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)    
        self._params = {}
        self._layers = OrderedDict()
        activation_layer = {'relu': ReLU}
        self.__init_weight(weight_init_std)
        for idx in range(1, self.hidden_layer_num+1):
            self._layers['Affine' + str(idx)] = Affine(self._params['W' + str(idx)],
                                                      self._params['b' + str(idx)])                                       
            self._params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
            self._params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
            self._layers['BatchNorm' + str(idx)] = BatchNormalization(self._params['gamma' + str(idx)], self._params['beta' + str(idx)])
                
            self._layers['Activation_function' + str(idx)] = activation_layer["relu"]()
            
        idx = self.hidden_layer_num + 1
        self._layers['Affine' + str(idx)] = Affine(self._params['W' + str(idx)], self._params['b' + str(idx)])

        self._last_layer = SoftMaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
            self._params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self._params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self,x):
        for layer in self._layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self._last_layer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t,axis=1)

        acc = np.sum(y == t) / float(x.shape[0])

        return acc
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self._last_layer.backward(dout)

        layers = list(self._layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self._layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self._layers['Affine' + str(idx)].db

            if idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self._layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self._layers['BatchNorm' + str(idx)].dbeta

        return grads