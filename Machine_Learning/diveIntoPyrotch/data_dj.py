import sys
import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def load_data_fashion_mnist(batch_size):
	mnist_train = torchvision.datasets.FashionMNIST(root='/document/2019/python/Data/',
                                                train=True, download=True, transform=transforms.ToTensor())
	mnist_test = torchvision.datasets.FashionMNIST(root='/document/2019/python/Data/', 
                                               train=False, download=True, transform=transforms.ToTensor())
	if sys.platform.startswith('win'):
	   	num_workers = 0 
	else:
		num_workers = 4
	train_iter = torch.utils.data.DataLoader(mnist_train, 
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
	test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_iter, test_iter

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step() # “softmax回归的简洁实现”一一节将用用到
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' 
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

def say(name):
    print(name, "you are well!")


def create_data(num_examples, num_inputs):
    '''生成随机数据
        @param num_examples 为生成数据的个数
        @param num_inputs 为数据的维度
    '''
    true_w = [2, -3.4]
    true_b = 4.2
    # 特征以及权重，b数据类型要一致
    # 随机生成1000个2维，范围在0-1之间的样本数据，转换为torch格式向量
    features = torch.from_numpy(np.random.normal(
        0, 1, (num_examples, num_inputs))).to(torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 标签数据随机增加0.01的误差
    labels += torch.from_numpy(np.random.normal(0,
                                                0.01, size=labels.size())).to(torch.float32)
    return features, labels

def linreg(X, w, b): # 矩阵相乘，前向传播
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): # 损失函数
    return (y_hat - y.view(y_hat.size()))** 2 / 2
    
def sgd(params, lr, batch_size): # 优化函数
    for param in params:
        param.data -= lr * param.grad / batch_size

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        # 最后一一次可能不不足足一一个batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def showData(x, y):
    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend(loc='upper left')
    plt.scatter(x, y, c='',
                        alpha=1.0, linewidth=1.0, marker='o', edgecolors='yellow',
                        s=55, label='test set')
    plt.show()

class FlattenLayer(nn.Module):
	def __init__(self):
		super(FlattenLayer, self).__init__()
	
	def forward(self, x): # x shape: (batch, *, *, ...)
		return x.view(x.shape[0], -1)