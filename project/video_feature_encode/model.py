
import torch
from torch import nn
from torch.nn import functional as F

## 使用多层感知机，数据量太大不能全部放入内存 #
class MLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512000, 25600),
            nn.ReLU(),
            # nn.Linear(256000, 102400),
            # nn.ReLU(),
            # nn.Linear(102400, 25600),
            # nn.ReLU(),
            # nn.Linear(51200, 25600),
            # nn.ReLU(),
            nn.Linear(25600, 5120),
            nn.ReLU(),
            # nn.Linear(10240, 5120),
            # nn.ReLU(),
            nn.Linear(5120, 1024),
            nn.ReLU(),
            # nn.Linear(2560, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.out = nn.Linear(256, 5)  # 输出层

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
# lenet网络
class LeNet(nn.Module):
    '''LeNet主要分为两部分
        1、卷积加池化，卷积层减小了尺寸增加了通道数，获取空间特征
        2、全连接层，将每个数据输出为一维数据，并逐渐减小个数
        3、LeNet未使用丢弃法
    '''
    def __init__(self, in_channel, out_channel):
        super(LeNet, self).__init__()
        self.conv2d_ksize = 5   
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 6, self.conv2d_ksize), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride

            nn.Conv2d(6, 16, self.conv2d_ksize),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, self.conv2d_ksize),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, self.conv2d_ksize),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(1364 * 64, 512 * 8),
            nn.ReLU(),
            nn.Linear(512 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_channel)
        )
    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)

        output = self.fc(feature)
        return output
    
if __name__ == "__main__":
    net = LeNet(1, 5)
    print(net.conv)
    X = torch.rand(size=(1, 1, 250, 2408), dtype=torch.float32)

    for layer in net.conv:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)
    print(X.shape)
    for layer in net.fc:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)
