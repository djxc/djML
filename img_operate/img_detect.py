# -*- coding: utf-8 -*- 
'''
@FileDesciption 图像识别
@Author small dj
@Date 2020-11-25
@LastEditor small dj
@LastEditTime 2020-11-25 19:41
'''
import sys
import torch
from torch import nn, optim
import numpy as np

from .dataset import MNISTData
from .operateIMG import hidden_layer
from .dj_model import DJModel

lr = 0.001
model_path = "/2020/result/weights_unet_mnist.pth"
run_type = "test"      # train or test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
maxpool = nn.MaxPool2d(2)
lastLayer = nn.Conv2d(128, 10, 1)

def createModel():
    '''生成模型，初始化损失函数以及优化函数'''
    model = DJModel(1, 10).to(device)

    if run_type == "test":
        load_model(model)
        detect_img(model)
    else:
        # crossEntropyLoss输入的label为真实label不能为one-hot编码
        criterion = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()                       # 损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化函数,并设置学习率
        train_model(model, criterion, optimizer)


def train_model(model, criterion, optimizer):
    train_loader, test_loader = MNISTData()
    index = 1
    epoch_loss = 0
    for epoch in range(3):
        if epoch > 0:
            adjust_lr(optimizer, epoch)
        for i, (images, labels) in enumerate(train_loader):
            # 将label转换为one-hot
            # newLabel = createOneHot(labels)        
            # print(newLabel.shape, images.shape)

            images = images.to(device)
            newLabel = labels.to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # print(outputs, newLabel)
            loss = criterion(outputs, newLabel)   # 损失函数
            loss.backward()                     # 后向传播
            optimizer.step()                    # 参数优化
            epoch_loss += loss.item()
            # print(i, images.shape, labels.shape, outputs.shape, newLabel.shape)

            index += 1
            if index % 30 == 0:
                # print(newLabel, outputs, loss.item())
                print(epoch, "     -----   ", index , "   ----   ", loss.item())
                # break
    save_model(model)
    print("save model...")

def detect_img(model):
    '''识别图片'''
    train_loader, test_loader = MNISTData()
    index = 1
    epoch_loss = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        # 前向传播
        outputs = model(images)
        print(outputs.argmax(axis=1), labels, outputs.argmax(axis=1).cpu() - labels)
        index += 1
        if index > 200:
            break

def adjust_lr(optimizer, epoch):
    '''动态更新学习率，每2个epoch将学习率减少为之前的10%'''
    lr_ = lr * (0.9 ** (epoch//2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def save_model(model):
    '''保存模型'''
    torch.save(model.state_dict(), model_path)

def load_model(model):
    '''模型加载参数'''
    print("load model...")
    model.load_state_dict(torch.load(model_path))  # 加载训练数据权重

def createOneHot(labels):
    '''生成one-hot编码
        利用numpy生成和类别数相同的一维数组，每个都为0，然后将对应类位置数修改为1
        最后将其转换为torch类型的tensor返回
    '''
    oneMetri = np.eye(10)
    oneHot = []
    for num in labels:
        index = num.item()
        label_ = oneMetri[index]
        oneHot.append(label_)
    # print(oneMetri[1], labels, labels.shape)
    # print(oneHot)
    # [index] = 1
    return torch.from_numpy(np.asarray(oneHot))

def convIMG(img_tensor, maxpool):
    '''卷积图像，将图像进行4次(卷积、归一化、激活)操作，每次操作跟一个最大池化'''
     # pool_img1 = conv_torch(img2tensor(img2))
    hidden_layer_img1 = hidden_layer(1, 16, img_tensor)
    hidden_layer_img1 = maxpool(hidden_layer_img1)

    hidden_layer_img2 = hidden_layer(16, 32, hidden_layer_img1)
    hidden_layer_img2 = maxpool(hidden_layer_img2)
    
    hidden_layer_img3 = hidden_layer(32, 64, hidden_layer_img2)
    hidden_layer_img3 = maxpool(hidden_layer_img3)

    hidden_layer_img4 = hidden_layer(64, 128, hidden_layer_img3)
    hidden_layer_img4 = maxpool(hidden_layer_img4)
    # 最后一层输出一个长度为10的数组
    hidden_layer_img = lastLayer(hidden_layer_img4)
    out = hidden_layer_img.reshape(10)
    # print(out)
    # 最后返回sigmoid函数处理过的值，将原本正负的值映射到0-1之间
    return nn.Sigmoid()(out)
