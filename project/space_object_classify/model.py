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


class SelfAttention(nn.Module):  
    def __init__(self, in_channels):  
        super(SelfAttention, self).__init__()  
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)  
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  
        self.gamma = nn.Parameter(torch.zeros(1))  
  
    def forward(self, x):  
        batch_size, width, height = x.size()  
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C  
        key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C x N  
        value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x N  
  
        attention = torch.bmm(query, key)  # B x N x N  
        attention = F.softmax(attention, dim=-1)  # B x N x N  
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N  
        out = out.view(batch_size, width, height)  # B x C x W x H  
  
        out = self.gamma * out + x  
        return out
    
class SEBlock(nn.Module):  
    def __init__(self, channel, reduction=16):  
        super(SEBlock, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(  
            nn.Linear(channel, channel // reduction, bias=False),  
            nn.ReLU(inplace=True),  
            nn.Linear(channel // reduction, channel, bias=False),  
            nn.Sigmoid()  
        )  
  
    def forward(self, x):  
        """通道注意力机制
            1、输入的图像是 [batch_size, channel, width, heigh]
            2、首先经过自适应平均池化层，输出是一维数据，然后reshape到batch_size x channel
        """
        b, c, _, _ = x.size()  
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1, 1)  
        return x * y.expand_as(x)  

class ResNet(nn.Module):
    def __init__(self, dropout_p=0.5):
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
        self.attention = SelfAttention(1024)
        self.se_block = SEBlock(1024)
        self.fc_fuse = nn.Linear(1024, 512)

        self.dropout = nn.Dropout(p=dropout_p)

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
        # out = self.global_avg_pool(out)

        sar = self.layer1_sar(sar_images)
        sar = self.resnet_block1_sar(sar)
        sar = self.resnet_block2_sar(sar)
        sar = self.resnet_block3_sar(sar)
        sar = self.resnet_block4_sar(sar)
        # sar = self.global_avg_pool_sar(sar)

        # # 分别提取可见光以及sar特征
        # # 提取可见光图像特征  
        # visible_features = self.extract_features(self.visible_cnn, visible_images)  
          
        # # 提取SAR图像特征  
        # sar_features = self.extract_features(self.sar_cnn, sar_images)  
        # sar_features = self.sar_fc(sar_features.view(sar_features.size(0), -1))  # 展平并映射到512维  
          
        # 融合特征,这里将sar和out作为不同的波段进行融合  
        # sar = sar.permute(0, 3, 2, 1) # sar.view(out.size(0), -1)
        # out = out.permute(0, 3, 2, 1) # out.view(out.size(0), -1)
        fused_features = torch.cat((out, sar), dim=1)  
        # fused_features = self.attention(fused_features)
        fused_features = self.se_block(fused_features) 
        fused_features = self.global_avg_pool(fused_features) 
        fused_features = fused_features.view(out.size(0), -1)
        fused_features = self.dropout(fused_features)

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
    

  
class SEBottleneck(nn.Module):  
    expansion = 4  
  
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):  
        super(SEBottleneck, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(out_channels)  
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,  
                               padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(out_channels)  
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  
        self.relu = nn.ReLU(inplace=True)  
        self.downsample = downsample  
        self.se = SEBlock(out_channels * self.expansion, reduction=reduction)  
  
    def forward(self, x):  
        identity = x  
  
        out = self.conv1(x)  
        out = self.bn1(out)  
        out = self.relu(out)  
  
        out = self.conv2(out)  
        out = self.bn2(out)  
        out = self.relu(out)  
  
        out = self.conv3(out)  
        out = self.bn3(out)  
  
        if self.downsample is not None:  
            identity = self.downsample(x)  
  
        out += identity  
        out = self.relu(out)  
        out = self.se(out)  
  
        return out  
  
class SEResNet(nn.Module):  
    def __init__(self, block, layers, num_classes=1000, reduction=16):  
        super(SEResNet, self).__init__()  
        self.in_channels = 64  
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)  
        self.relu = nn.ReLU(inplace=True)  
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=reduction)  
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)  
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=reduction)  
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=reduction)  
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc_cate = nn.Linear(512 * block.expansion, num_classes)  
        self.fc_ = nn.Linear(512 * block.expansion, num_classes)  
        self.fc_cate = nn.Linear(512 * block.expansion, num_classes)  
        self.fc_cate = nn.Linear(512 * block.expansion, num_classes)  
  
        for m in self.modules():  
            if isinstance(m, nn.Conv2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)  
  
    def _make_layer(self, block, out_channels, blocks, stride=1, reduction=16):  
        downsample = None  
        if stride != 1 or self.in_channels != out_channels * block.expansion:  
            downsample = nn.Sequential(  
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(out_channels * block.expansion),  
            )  
  
        layers = []  
        layers.append(block(self.in_channels, out_channels, stride, downsample, reduction=reduction))  
        self.in_channels = out_channels * block.expansion  
        for _ in range(1, blocks):  
            layers.append(block(self.in_channels, out_channels, reduction=reduction))  
  
        return nn.Sequential(*layers)  
  
    def forward(self, x):  
        x = self.conv1(x)  
        x = self.bn1(x)  
        x = self.relu(x)  
        x = self.maxpool(x)  
  
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)  
  
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
  
        return x  
  
# Example usage  
def seresnet18():  
    return SEResNet(SEBottleneck, [2, 2, 2, 2])  

    

