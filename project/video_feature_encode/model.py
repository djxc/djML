
import os 
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import timm

from config import workspace_root

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
class OldLeNet(nn.Module):
    '''LeNet主要分为两部分
        1、卷积加池化，卷积层减小了尺寸增加了通道数，获取空间特征
        2、全连接层，将每个数据输出为一维数据，并逐渐减小个数
        3、LeNet未使用丢弃法
    '''
    def __init__(self, in_channel, out_channel):
        super(OldLeNet, self).__init__()
        self.conv2d_ksize = 5   
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 6, self.conv2d_ksize), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride

            nn.Conv2d(6, 16, self.conv2d_ksize),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Conv2d(16, 32, self.conv2d_ksize),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),

            # nn.Conv2d(32, 64, self.conv2d_ksize),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),           
        )
        self.fc = nn.Sequential(
            nn.Linear(10912 * 8, 512 * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512 * 8, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, out_channel)
        )
    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

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
            nn.Linear(10912 * 8, 512 * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512 * 8, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, out_channel)
        )
    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# lenet网络
class LeNetBN(nn.Module):
    '''LeNet主要分为两部分
        1、卷积加池化，卷积层减小了尺寸增加了通道数，获取空间特征
        2、全连接层，将每个数据输出为一维数据，并逐渐减小个数
        3、LeNet未使用丢弃法
    '''
    def __init__(self, in_channel, out_channel):
        super(LeNetBN, self).__init__()
        self.conv2d_ksize = 5   
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 6, self.conv2d_ksize), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride

            nn.Conv2d(6, 16, self.conv2d_ksize),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, self.conv2d_ksize),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, self.conv2d_ksize),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           
        )
        self.fc = nn.Sequential(
            nn.Linear(10912 * 8, 512 * 8),
            nn.BatchNorm1d(512*8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, out_channel)
        )
    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# alexnet网络
class AlexNet(nn.Module):
    '''LeNet主要分为两部分
        1、卷积加池化，卷积层减小了尺寸增加了通道数，获取空间特征
        2、全连接层，将每个数据输出为一维数据，并逐渐减小个数
        3、LeNet未使用丢弃法
    '''
    def __init__(self, in_channel, out_channel):
        super(AlexNet, self).__init__()
        self.conv2d_ksize = 3   
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

            nn.Conv2d(64, 128, self.conv2d_ksize),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, self.conv2d_ksize),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(960 * 8, 512 * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512 * 8, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, out_channel)
        )
    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output
    
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
  
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels, class_num) -> None:
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, nn.Linear(1024, class_num))
    
    def forward(self, x):
        return self.net(x)

def create_net(net_name: str, class_num: int, resume=""):
    """根据模型名称创建模型
    """
    print("create {} net ....".format(net_name))
    if net_name == "efficientNet":
        pass
        # net = EfficientNet.from_pretrained('efficientnet-b4',  num_classes=class_num)
        # net = EfficientNet.from_name('efficientnet-b4',  num_classes=176)
    # elif net_name == "resNet":
    #     net = createResNet()
    elif net_name == "resNet50_pre":
        net = torchvision.models.resnet50(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # set_parameter_requires_grad(model_ft, False)  # 固定住前面的网络层
        num_ftrs = net.fc.in_features
        # 修改最后的全连接层
        net.fc = nn.Sequential(
            nn.Linear(num_ftrs, class_num)
        )
    elif net_name == "resNet101_pre":
        net = torchvision.models.resnet101(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # set_parameter_requires_grad(model_ft, False)  # 固定住前面的网络层
        num_ftrs = net.fc.in_features
        # 修改最后的全连接层
        net.fc = nn.Sequential(
            nn.Linear(num_ftrs, class_num)
        )
    elif net_name == "resnet50_pre_timm":
        net = timm.create_model("resnet50d", pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # for i, param in enumerate(net.parameters()): 
        #     param.requires_grad = False
        #     if i == 160:
        #         param.requires_grad = True
        n_features = net.fc.in_features
        net.fc = nn.Linear(n_features, class_num)
        # net.fc.requires_grad_ = True
    elif net_name == "resNet18_pre":
        net = torchvision.models.resnet18(pretrained=True)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # set_parameter_requires_grad(model_ft, False)  # 固定住前面的网络层
        # for i, param in enumerate(net.parameters()): 
        #     if i == 0:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        num_ftrs = net.fc.in_features
        # 修改最后的全连接层
        net.fc = nn.Sequential(
            nn.Linear(num_ftrs, class_num),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(64, class_num)
        )
    elif net_name == "resnext":
        net = torchvision.models.resnext50_32x4d(pretrained=True)
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(num_ftrs, class_num))
    elif net_name == "alexNet":
        net = AlexNet(1, class_num)
    elif net_name == "leNet":
        net = LeNet(1, class_num)
    elif net_name == "leNet_bn":
        net = LeNetBN(1, class_num)
    else:
        net = LeNet(1, class_num)

    if resume and len(resume) > 0:
        net.load_state_dict(torch.load(os.path.join(workspace_root, resume)))        # 加载训练数据权重
        print("load model {}".format(resume))
    return net

if __name__ == "__main__":
    net = LeNet(1, 5)
    print(net.conv)
    X = torch.rand(size=(1, 1, 250, 2048), dtype=torch.float32)

    for layer in net.conv:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)
    print(X.shape)
    for layer in net.fc:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)
    # net = create_net("resnet50_pre_timm", 5, None)
    # print(net)
