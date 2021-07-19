# 学习多层感知机

import torch
import numpy as np
import sys

import data
from util import train_ch3
from model import FlattenLayer


batch_size = 256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 隐含层权重以及偏移量
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), 
                dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)),
                dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)

train_iter, test_iter = data.load_data_fashion_mnist(batch_size)

def relu(X):
    '''relu激活函数，大于0返回本身，否则返回0'''
    return torch.max(input=X, other=torch.tensor(0.0))

def net(X):
    '''多层感知机，多层感知机如果不添加非线性激活函数与单层网络没有什么区别
        1、首先将数据压缩为1维
        2、数据与第一层权重相乘加偏移量，在用激活函数得到第一层的输出
        3、第二层将第一层的输出与第二层的权重相乘，然后添加偏移量
    '''
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

def multilayerPerceptron():

    loss = torch.nn.CrossEntropyLoss()
    num_epochs, lr = 5, 100.0
    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, [W1, b1, W2, b2], lr)


def multilayerPerceptronTorch():
    from torch import nn
    from torch.nn import init
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )
    # 初始化模型参数
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
        batch_size = 256
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    num_epochs = 5
    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

if __name__ == "__main__":
    # multilayerPerceptron()
    multilayerPerceptronTorch()