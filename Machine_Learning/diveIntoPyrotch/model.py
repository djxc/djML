# 定义模型
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from util import multibox_prior


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗⼝口形状设置成输⼊入的⾼高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使⽤用传⼊入的移动平均所得的均值和⽅方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使⽤用全连接层的情况，计算特征维上的均值和⽅方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使⽤用⼆二维卷积层的情况，计算通道维上（axis=1）的均值和⽅方差。这⾥里里我们需要保持
            # X的形状以便便后⾯面可以做⼴广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2,
            keepdim=True).mean(dim=3, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=0,
        keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下⽤用当前的均值和⽅方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更更新移动平均的均值和⽅方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 拉伸和偏移
    return Y, moving_mean, moving_var

# 批量归一化，利用小批量上的均值和标准差，不断调整神经网络的中间输出，从而使整个神经网络在各层中间值数值更稳定
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
            # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))
            # 不不参与求梯度和迭代的变量量，全在内存上初始化成0
            self.moving_mean = torch.zeros(shape)
            self.moving_var = torch.zeros(shape)
    def forward(self, X):
        # 如果X不不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更更新过的moving_mean和moving_var, Module实例例的traning属性默认为true, 调⽤用.eval()后设成false
            Y, self.moving_mean, self.moving_var = batch_norm(self.training,
                X, self.gamma, self.beta, self.moving_mean,
                self.moving_var, eps=1e-5, momentum=0.9)
        return Y


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
        1、卷积加池化，卷积层减小了尺寸增加了通道数，获取空间特征
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

# VGG网络类似于AlexNet与LeNet，都为卷积层后跟全连接层，卷积模块为多个VGG块组成
def VGG_Block(num_convs, in_channels, out_channels):
    '''VGG块
        1、vgg即为将简单的vgg块叠加，构建深层网络
        2、vgg块每一卷积层后跟一个ReLU激活函数。第一卷积层输入为输入的通道数，其他卷积层为输入输出都为块输出层。
        3、每vgg块最后添加最大池化层
        4、数据每经过一个VGG块，数据通道数会增加，高宽会减半
    '''

    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))   # 使高宽减半
    return nn.Sequential(*blk)


def VGG(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1), VGG_Block(num_convs, in_channels, out_channels))
    # 全连接层
    net.add_module("fc", nn.Sequential(
        FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10),
        ))
    return net


# 随着网络加深，loss不降反升
# resnet则将上一块的输出与输入进行运算，作为本次的输入，这样会保留更多的信息，不至于输入携带的信息太少引起loss下降
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第⼀一个模块的通道数同输⼊入通道数⼀一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def createResNet():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) #GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))
    return net

# 稠密网络DenseNet，稠密网络类似于残差，都是实现跨层连接，不同的是跨层的输入与输出的连接方式，
# 残差为输入与输出相加，稠密为将输入与输出进行拼接
# 输入与输出拼接会导致数据通道数以及长宽都增加，需要引入过渡层将通道与长宽减半
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
    nn.ReLU(),
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    # 计算输出通道数
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 在通道维上将输⼊入和输出连结
        return X

def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
    return blk


def createDenseNet():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    num_channels, growth_rate = 64, 32 # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs, num_channels, growth_rate)
        net.add_module("DenseBlosk_%d" % i, DB)
        # 上⼀一个稠密块的输出通道数
        num_channels = DB.out_channels
        # 在稠密块之间加⼊入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            net.add_module("transition_block_%d" % i, transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2
    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", GlobalAvgPool2d()) #GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(num_channels, 10)))
    return net



# 全卷积网络：利用卷积层对图像进行特征提取，
# 然后利用1x1卷积层将通道数变换为类别个数，
# 最后通过转置卷积层将特征图的高和宽变换为输入图像的尺寸
# 转置矩阵用用双线性内插进行实现，目的为上采样
def createFCN(num_classes = 21):
    '''创建全卷积网络
        1、这里以残差网络作为主干，去除最后两层
        @param num_classes 需要识别的类个数
    '''
    # resNet = createResNet()
    resNet = torchvision.models.resnet18(pretrained=False)
    net = nn.Sequential(*list(resNet.children())[:-2])
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module(
        'transpose_conv',
        nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16,
                       stride=32))

    conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
    # 用双线性插值的上采样初始化转置卷积层
    conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)
    return net

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros(
        (in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


