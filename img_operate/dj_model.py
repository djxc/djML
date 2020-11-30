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

        # 池化层不会修改通道数，会改变每个通道尺寸，步幅默认与卷积核大小一致，如这里的2表示，
        # 将2*2数据取最大的值作为新的数据，然后移动步幅大小，重复以上步骤。这里的2会将图像尺寸缩小为原来的一半
        self.maxpool = nn.MaxPool2d(2)

        # 最后一层为线性，将高纬度转化为低维度
        self.lastLayer = nn.Linear(128, out_ch)

        self.hiddenLayer1 = self.hidden_layer(self.in_ch, 16)    
        self.hiddenLayer2 = self.hidden_layer(16, 32)        
        self.hiddenLayer3 = self.hidden_layer(32, 64)       
        self.hiddenLayer4 = self.hidden_layer(64, 128)      

    def forward(self, img_tensor):
        '''组织init中定义的运算，进行前向传播
        卷积图像，将图像进行4次(卷积、归一化、激活)操作，每次操作跟一个最大池化
        '''
        hidden_layer_img1 = self.hiddenLayer1(img_tensor)
        hidden_layer_img1 = self.maxpool(hidden_layer_img1)

        hidden_layer_img2 = self.hiddenLayer2(hidden_layer_img1)
        hidden_layer_img2 = self.maxpool(hidden_layer_img2)
        
        hidden_layer_img3 = self.hiddenLayer3(hidden_layer_img2)
        hidden_layer_img3 = self.maxpool(hidden_layer_img3)

        hidden_layer_img4 = self.hiddenLayer4(hidden_layer_img3)
        hidden_layer_img4 = self.maxpool(hidden_layer_img4)

        # 最后一层输出一个长度为10的数组,将数组扁平化
        out = self.lastLayer(hidden_layer_img4.squeeze())       

        # softmax为将不同类的概率之和为1，logSoftmax则在softmax基础上增加了log
        # sigmoid函数处理过的值，将原本正负的值映射到0-1之间，对二分类适合
        # dim=0表示对列元素进行运算，dim=1表示对行元素进行运算
        out = nn.Softmax(dim=1)(out)
        # out = nn.Sigmoid()(out)
        # out = nn.LogSoftmax(dim=1)(out)
        return out


    def hidden_layer(self, in_ch, out_ch):
        '''采用Squential将多个操作包裹在一起成为一个步骤，包括两步卷积、归一化以及激活函数
            Conv2d卷积层只改变通道数不改变每个通道尺寸，会将输入数据的通道数改变为输出通道数，每个卷积层都会有一个卷积核需要训练
        '''
        hiddenLayer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return hiddenLayer