import torch
import numpy as np
import matplotlib.pyplot as plt


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
