# coding: utf-8
import numpy as np

def relu(x):
    return np.max(0, x)

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad