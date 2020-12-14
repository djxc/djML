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

from .datasets.dataset import MNISTData
from .operateIMG import hidden_layer
from .models.dj_model import DJModel
from .loss import cross_entropy_loss

lr = 0.001
model_path = "/2020/result/weights_unet_mnist.pth"
run_type = "test"      # train or test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
    '''训练模型'''
    train_loader, test_loader = MNISTData()
    index = 1
    epoch_loss = 0
    for epoch in range(3):
        if epoch > 0:
            adjust_lr(optimizer, epoch)
        for i, (images, labels) in enumerate(train_loader):
            # 将label转换为one-hot
            labels = createOneHot(labels)        

            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # loss = criterion(outputs, labels)  # 损失函数
            loss = cross_entropy_loss(outputs, labels)   # 损失函数            
            loss.backward()  # 后向传播
            optimizer.step()                    # 参数优化
            epoch_loss += loss.item()

            index += 1
            if index % 50 == 0:
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
        if torch.sum(outputs.argmax(axis=1).cpu() - labels).item() != 0:
            epoch_loss += 1
        index += 1
        if index > 400:
            print(1-epoch_loss/400)
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
    return torch.from_numpy(np.asarray(oneHot))

