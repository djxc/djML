"""
UNet网络
"""

import torch
from torch import nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet1, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = torch.sigmoid(out)
        return out

"""unet双卷积块，两次：卷积 + batchNorm + relu"""
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        # 将对应的下采样层合并，保留位置信息
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)

        merge8 = torch.cat([up_8, c2], dim=1)

        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)

        merge9 = torch.cat([up_9, c1], dim=1)
        
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        # sigmoid将值转换为概率
        out = nn.Sigmoid()(c10)        
        return out
    
    def forward1(self, x):
        print("    X -> ", x.shape)
        c1 = self.conv1(x)
        print("    c1 -> ", c1.shape)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        print("    c2 -> ", c2.shape)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        print("    c3 -> ", c3.shape)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        print("    c4 -> ", c4.shape)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)
        print("    c5 -> ", c5.shape)

        up_6 = self.up6(c5)
        print("  up_6 -> ", up_6.shape)
        merge6 = torch.cat([up_6, c4], dim=1)
        print("up6_c4 -> ", merge6.shape)
        c6 = self.conv6(merge6)
        print("    c6 -> ", c6.shape)

        up_7 = self.up7(c6)
        print("  up_7 -> ", up_7.shape)
        merge7 = torch.cat([up_7, c3], dim=1)
        print("up7_c3 -> ", merge7.shape)
        c7 = self.conv7(merge7)
        print("    c7 -> ", c7.shape)

        up_8 = self.up8(c7)
        print("   up8 -> ", up_8.shape)

        merge8 = torch.cat([up_8, c2], dim=1)
        print("up8_c2 -> ", merge8.shape)

        c8 = self.conv8(merge8)
        print("    c8 -> ", c8.shape)

        up_9 = self.up9(c8)
        print("   up9 -> ", up_9.shape)

        merge9 = torch.cat([up_9, c1], dim=1)
        print("up9_c1 -> ", merge9.shape)
        
        c9 = self.conv9(merge9)
        print("    c9 -> ", c9.shape)

        c10 = self.conv10(c9)
        print("   c10 -> ", c10.shape)
        print(c10)
        #out = nn.Sigmoid()(c10)
        return c10
