'''
@FileDesciption 利用pytorch框架定义自己的模型
@Author small dj
@Date 2020-11-25
@LastEditor small dj
@LastEditTime 2020-11-25 19:41
'''
import torch
from torch import nn


class DJModel(nn.Module):
    '''继承torch下的Module'''
    def __init__(self, in_ch, out_ch):
        '''init中定义一些运算'''
        super(DJModel, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.maxpool = nn.MaxPool2d(2)
        # 最后一层为线性，将高纬度转化为低维度
        self.lastLayer = nn.Linear(128, out_ch)
        self.hiddenLayer1 = nn.Sequential(
            nn.Conv2d(self.in_ch, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.hiddenLayer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.hiddenLayer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.hiddenLayer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_tensor):
        '''组织init中定义的运算，进行前向传播'''

        '''卷积图像，将图像进行4次(卷积、归一化、激活)操作，每次操作跟一个最大池化'''
        hidden_layer_img1 = self.hiddenLayer1(img_tensor)
        hidden_layer_img1 = self.maxpool(hidden_layer_img1)

        hidden_layer_img2 = self.hiddenLayer2(hidden_layer_img1)
        hidden_layer_img2 = self.maxpool(hidden_layer_img2)
        
        hidden_layer_img3 = self.hiddenLayer3(hidden_layer_img2)
        hidden_layer_img3 = self.maxpool(hidden_layer_img3)

        hidden_layer_img4 = self.hiddenLayer4(hidden_layer_img3)
        hidden_layer_img4 = self.maxpool(hidden_layer_img4)
        # print(hidden_layer_img4.shape)
        # 最后一层输出一个长度为10的数组
        out = self.lastLayer(hidden_layer_img4.squeeze())
        # print(out)
        # print(hidden_layer_img.shape,  hidden_layer_img.squeeze())
        # out = hidden_layer_img.reshape(self.out_ch)
        # out = hidden_layer_img.squeeze()
        # out = nn.Softmax(dim=1)(out)
        out = nn.LogSoftmax(dim=1)(out)
        # print(out)

        # 最后返回sigmoid函数处理过的值，将原本正负的值映射到0-1之间，对二分类适合
        # 对多分类来说可以使用softmax将多个类的概率之和为1
        return out # nn.Sigmoid()(out)


    def hidden_layer(self, in_ch, out_ch, img):
        '''采用Squential将多个操作包裹在一起成为一个步骤，包括两步卷积、归一化以及激活函数'''
        hiddenLayer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return hiddenLayer(img)