from __future__ import print_function, division

from .models.insectDetectNet import InsectDetectNet
from .models.vgg16 import vgg16_bn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '/2020/data/hymenoptera_data'


def insectData():
    '''加载昆虫数据，并进行数据增强'''
    data_transforms = {
        # 训练中的数据增强和归一化
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # 随机裁剪
            transforms.RandomHorizontalFlip(), # 左右翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值方差归一化
        ]), 
        # 验证集不增强，仅进行归一化
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

def imshow(inp, title=None):
    # 将输入的类型为torch.tensor的图像数据转为numpy的ndarray格式
    # 由于每个batch的数据是先经过transforms.ToTensor()函数从numpy的ndarray格式转换为torch.tensor格式，这个转换主要是通道顺序上做了调整：
    # 由原始的numpy中的BGR顺序转换为torch中的RGB顺序
    # 所以我们在可视化时候，要先将通道的顺序转换回来，即从RGB转回BGR
    plt.ion()
    inp = inp.numpy().transpose((1, 2, 0))
    # 接着再进行反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)

def seeIMG(dataloaders, class_names):
    #  从训练数据中取一个batch的图片
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每一个epoch都会进行一次验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            running_loss = 0.0
            running_corrects = 0

            #  迭代所有样本
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 将梯度归零
                optimizer.zero_grad()

                # 前向传播网络，仅在训练状态记录参数的梯度从而计算loss
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播来进行梯度下降
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计loss值
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 依据验证集的准确率来更新最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 载入最优模型
    model.load_state_dict(best_model_wts)
    return model

def train():
    dataloaders, dataset_sizes, class_names = insectData()
    # net = InsectDetectNet().to(device)
    # net, optimizer_ft, exp_lr_scheduler = res18model()
    # net = vgg16_bn(pretrained=True, num_classes=2).to(device)
    net, optimizer_ft, exp_lr_scheduler = vgg16Model()
    # print(net)

    # 定义分类loss 
    criterion = nn.CrossEntropyLoss()

    # 优化器使用sgd，学习率设置为0.001
    # optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 每7个epoch将lr降低为原来的0.1
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 进行训练
    cnn_model = train_model(net, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=25) 

def vgg16Model():
    '''
        采用训练好的resnet18模型进行训练
        1、需要修改最后输出的维数，修改最后一层全连接层
        2、模型训练只更新全连接层的参数
    '''
    net = vgg16_bn(pretrained=True).to(device)
    
    # freeze前面的卷积层，使其训练时不更新
    for param in net.parameters():
        param.requires_grad = False

    # 最后的分类fc层输出换为2，进行二分类
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, 2)

    net = net.to(device)

    # 仅训练最后改变的fc层
    optimizer_conv = optim.SGD(net.classifier[6].parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    return net, optimizer_conv, exp_lr_scheduler

def res18model():
    '''
        采用训练好的resnet18模型进行训练
        1、需要修改最后输出的维数，修改最后一层全连接层
        2、模型训练只更新全连接层的参数
    '''
    # 从torchvision中载入resnet18模型，并且加载预训练
    model_conv = torchvision.models.resnet18(pretrained=True)
    # freeze前面的卷积层，使其训练时不更新
    for param in model_conv.parameters():
        param.requires_grad = False

    # 最后的分类fc层输出换为2，进行二分类
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    # 仅训练最后改变的fc层
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    return model_conv, optimizer_conv, exp_lr_scheduler

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)