import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗⼝口形状设置成输⼊入的⾼高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

# 将数据压缩为一维
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

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
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d()) #GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 176)))
    return net


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet_block1 = resnet_block(64, 64, 2, first_block=True)
        self.resnet_block2 = resnet_block(64, 128, 2)
        self.resnet_block3 = resnet_block(128, 256, 2)
        self.resnet_block4 = resnet_block(256, 512, 2)
        self.global_avg_pool = GlobalAvgPool2d() #GlobalAvgPool2d的输出: (Batch, 512, 1, 1)


        self.layer1_sar = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet_block1_sar = resnet_block(64, 64, 2, first_block=True)
        self.resnet_block2_sar = resnet_block(64, 128, 2)
        self.resnet_block3_sar = resnet_block(128, 256, 2)
        self.resnet_block4_sar = resnet_block(256, 512, 2)
        self.global_avg_pool_sar = GlobalAvgPool2d() #GlobalAvgPool2d的输出: (Batch, 512, 1, 1)

        # # 使用预训练的ResNet模型作为特征提取器（针对可见光图像）  
        # self.visible_cnn = models.resnet18(pretrained=True)  
        # self.visible_cnn.fc = nn.Identity()  # 移除原始的全连接层  
        # # 为SAR图像定义一个简单的CNN（或使用其他预训练模型）  
        # # 注意：这里为了简化，我们使用与可见光图像相同的结构，但通常SAR图像可能需要不同的架构  
        # self.sar_cnn = nn.Sequential(  
        #     *list(self.visible_cnn.children())[:-2]  # 复制ResNet的大部分层，但移除最后的平均池化和全连接层  
        # )  
        # self.sar_fc = nn.Linear(self.sar_cnn[-1][-1].bn2.num_features, 512)  # 假设我们需要将SAR特征映射到512维  
        self.fc_fuse = nn.Linear(1024, 512)

        self.fc1 = nn.Sequential(FlattenLayer(), nn.Linear(512, 10))
        self.fc2 = nn.Sequential(FlattenLayer(), nn.Linear(512, 2))
        self.fc3 = nn.Sequential(FlattenLayer(), nn.Linear(512, 2))
        self.fc4 = nn.Sequential(FlattenLayer(), nn.Linear(512, 3))

    def forward(self, visible_images, sar_images):
        out = self.layer1(visible_images)
        out = self.resnet_block1(out)
        out = self.resnet_block2(out)
        out = self.resnet_block3(out)
        out = self.resnet_block4(out)
        out = self.global_avg_pool(out)

        sar = self.layer1_sar(sar_images)
        sar = self.resnet_block1_sar(sar)
        sar = self.resnet_block2_sar(sar)
        sar = self.resnet_block3_sar(sar)
        sar = self.resnet_block4_sar(sar)
        sar = self.global_avg_pool_sar(sar)

        # # 分别提取可见光以及sar特征
        # # 提取可见光图像特征  
        # visible_features = self.extract_features(self.visible_cnn, visible_images)  
          
        # # 提取SAR图像特征  
        # sar_features = self.extract_features(self.sar_cnn, sar_images)  
        # sar_features = self.sar_fc(sar_features.view(sar_features.size(0), -1))  # 展平并映射到512维  
          
        # 融合特征  
        sar = sar.view(out.size(0), -1)
        out = out.view(out.size(0), -1)
        fused_features = torch.cat((out, sar), dim=1)  
        fused_features = F.relu(self.fc_fuse(fused_features))  

        out1 = self.fc1(fused_features)
        out2 = self.fc2(fused_features)
        out3 = self.fc3(fused_features)
        out4 = self.fc4(fused_features)
        return out1, out2, out3, out4
    
    def extract_features(self, model, x):  
        # 提取卷积网络的特征图，并应用全局平均池化  
        features = model(x)  
        features = F.adaptive_avg_pool2d(features, (1, 1))  
        features = features.view(features.size(0), -1)  
        return features  
    

