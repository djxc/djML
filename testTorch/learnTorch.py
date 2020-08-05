import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def test():
    net = Net()
    print(net)


def demo():
    # x是直接创建的没有grad_fn，只有通过运算获得的tensor才会有grad_fn
    x = torch.ones(2, 2, requires_grad=True)
    print(x)
    print(x.grad_fn)

    y = x + 4
    print(y, y.grad_fn)
    z = y * y * 3
    out = z.mean()
    print(z, out)
    out.backward()
    print(x.grad)


def showData(x, y):
    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend(loc='upper left')
    plt.scatter(x, y, c='',
                        alpha=1.0, linewidth=1.0, marker='o', edgecolors='yellow',
                        s=55, label='test set')
    plt.show()


def dataSet(batchSize, features, labels):
    '''批量读取，每次读取一个batch
        1、获取样本总数，生成样本的标号indices，将标号打乱，
    每次返回batchSize个样本与标签。最后一个可能不足batchSize个
    '''
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batchSize):
        j = torch.LongTensor(indices[i:min(i + batchSize, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def linearRegression():
    '''线性回归'''
    true_w = [2, -3.4]
    true_b = 4.2
    # 特征以及权重，b数据类型要一致
    features = torch.from_numpy(np.random.normal(0, 1, (1000, 2))).to(torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())).to(torch.float32)
    print(features[0], labels[0])
    # showData(features[:, 1], labels)
    batch_size = 10
    # 初始化模型的参数
    w = torch.tensor(np.random.normal(0, 0.01, (2, 1)),
                     dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linregModel
    loss = squared_loss
    for epoch in range(num_epochs):
        for x, y in dataSet(batch_size, features, labels):
            # 通过模型的前向传播计算预测值，然后利用损失函数计算与真实值之间的差距，
            # 然后对损失函数进行反向传播，利用优化函数更新权重等参数；最后需要将参数的导数归0
            l = loss(net(x, w, b), y).sum()
            l.backward()
            sgd([w, b], lr, batch_size)
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    print(w, b)

def linregModel(x, w, b):
    '''定义线性回归模型'''
    return torch.mm(x, w) + b


def squared_loss(y_hat, y):
    '''定义损失函数'''
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    '''优化函数'''
    for param in params:
        param.data -= lr * param.grad / batch_size

if __name__ == "__main__":
    # 矩阵运算要比遍历循环运算更快
    a = torch.ones(1000)
    b = torch.ones(1000)
    start = time.time()
    c = torch.zeros(1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    spend = time.time() - start
    print(spend)
    start1 = time.time()
    c = a + b
    print(time.time() - start1)
    linearRegression()
