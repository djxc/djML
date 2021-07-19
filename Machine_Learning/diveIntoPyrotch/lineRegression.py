# 线性回归
# 1、损失函数，如平方损失
# 2、优化算法
#

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

import data
import model
import loss as Loss

def LineRegressionBasic():
    '''利用原生tensor编写线性函数'''
    # 自己的包文件data_dj.py在该文件的当前目录下
    features, labels = data.create_data(1000, 2)
    print(features[0], labels[0])
    data.showData(features[:, 1], labels, True)

    # 模型训练过程
    batch_size = 30

    # 初始化权重以及偏移量
    w = torch.tensor(np.random.normal(0, 0.01, (2, 1)),
                        dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    lr = 0.03                        # 学习率
    num_epochs = 10                  # 训练循环次数
    net = model.linreg             # 自定义的模型
    loss = Loss.squared_loss      # 平方损失函数

    for epoch in range(num_epochs):
        for x, y in data.data_iter(batch_size, features, labels):
            # 通过模型的前向传播计算预测值，然后利用损失函数计算与真实值之间的差距，
            # 然后对损失函数进行反向传播，利用优化函数更新权重等参数；最后需要将参数的导数归0
            l = loss(net(x, w, b), y).sum()     # l需要进行sum运算将其转换为一个标量才可进行反向求导，否则tensor不可进行反向求导
            l.backward()
            data.sgd([w, b], lr, batch_size)
            w.grad.data.zero_()
            b.grad.data.zero_()
        # 每次循环之后，统计下loss情况
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    print(w, b)


def LineRegressionTorch():
    '''利用torch编写线性回归'''
    import torch.utils.data as Data
    import torch.nn as nn
    from torch.nn import init
    import torch.optim as optim
    features, labels = data.create_data(1000, 2)
    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 随机读取小小批量量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    net = model.LinearNet(2)
    # 初始化参数
    # init.normal_(net[0].weight, mean=0, std=0.01)
    # init.constant_(net[0].bias, val=0)
    # 损失函数以及优化函数
    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 15
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad() # 梯度清零,等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))

if __name__ == "__main__":
    LineRegressionTorch()
