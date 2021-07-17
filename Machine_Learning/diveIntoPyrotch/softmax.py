# 利用fashion-mnist数据集测试softmax


import torch
import torch.nn as nn
import torchvision
import numpy as np
import sys

import data
from loss import cross_entropy
from model import SoftMaxNet, SoftMaxNetTorch
from util import train_ch3

num_epochs, lr = 5, 0.1

def softmaxBasic():
    '''手动实现softmax'''
    batch_size = 256
    train_iter, test_iter = data.load_data_fashion_mnist(batch_size)
    num_inputs = 784
    num_outputs = 10
    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)    
    net = SoftMaxNet(num_inputs, W, b).run
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)



def softmaxTorch():
    ''''''
    batch_size = 256
    num_inputs = 784
    num_outputs = 10
    # 加载数据，
    train_iter, test_iter =data.load_data_fashion_mnist(batch_size)
    # 指定损失函数以及优化函数
    loss = nn.CrossEntropyLoss()
    net = SoftMaxNetTorch(num_inputs, num_outputs)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

if __name__ == "__main__":
    softmaxBasic()
    # softmaxTorch()