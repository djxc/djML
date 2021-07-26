# 定义模型
import torch
import torch.nn as nn


def linreg(X, w, b):  # 矩阵相乘，前向传播
    '''线性模型  
        1、返回矩阵与w乘积加上偏移量b  
        @param X 训练数据矩阵  
        @param w 权重  
        @param b 偏移量 
    '''
    return torch.mm(X, w) + b


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
        
    def forward(self, x):
        y = self.linear(x)
        return y

# 将数据压缩为一维
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

# 定义模型
class SoftMaxNetTorch(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(SoftMaxNetTorch, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
    

class SoftMaxNet():
    def __init__(self, num_inputs, W, b):
        self.num_inputs = num_inputs
        self.W = W
        self.b = b

    def softmax(self, X):
        '''softmax
            1、首先对数据进行exp函数运算，然后同一行求和，
        每个元素除以该行的和。得到元素非负且和为1
        '''
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        return X_exp / partition # 这里里里应用用了了广广播机制

    def run(self, X):
        '''将数据拍平，压缩为一行，与权重相乘，加上偏移量'''
        return self.softmax(torch.mm(X.view((-1, self.num_inputs)), self.W) + self.b)

# lenet网络
class LeNet(nn.Module):
    '''LeNet主要分为两部分
        1、卷积加池化，卷积层减小了尺寸增加了通道数
        2、全连接层，将每个数据输出为一维数据，并逐渐减小个数
        3、LeNet未使用丢弃法
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


# AlexNet 模型
class AlexNet(nn.Module):
    '''AlexNet模型与LeNet类似，包括卷积层与全连接层
        1、AlexNet层数更多
        2、AlexNet采用Relu激活函数，可以缓解梯度为0、过拟合问题
        3、全连接层采用dropout
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口,使用填充为2来使得输入与输出的高和宽一致,且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层,且使用更小的卷积窗口。除了最后的卷积层外,进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
            )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里里里使用用Fashion-MNIST,所以用用类别数为10,而而非非论文文中的1000
            nn.Linear(4096, 10),
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        ''''''
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)