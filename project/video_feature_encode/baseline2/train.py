#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 设置超参数
num_classes = 5
batch_size = 32
learning_rate = 0.001
num_epochs = 50

#####################################################


class linear50(nn.Module):
    # 定义模型
    def __init__(self, input_size=2048*50, num_classes=5):
        super().__init__()

        self.name = "linear50"

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class VideoDataSet(Dataset):
    # 定义数据集类
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.data)


def read_data(video_root_path):
    print("load")
    # data load
    with open(os.path.join(video_root_path, 'train', 'train_list.txt'), 'r') as f:
        train_list = json.load(f)

    train_count = len(train_list)
    train_input = torch.randn(train_count*5, 50, 2048)
    train_label = torch.arange(0, train_count*5)
    index = 0
    for video in tqdm(train_list):
        np_data_path = os.path.join(
            video_root_path, 'train', 'train_feature', video)
        np_data_250 = np.squeeze(np.load(np_data_path))

        # 50帧一组数据
        for i in range(0, 250, 50):
            np_data_50 = np_data_250[i:i+50]
            train_input[index, :] = torch.from_numpy(np_data_50)
            train_label[index] = float(train_list[video])
            index += 1

    train_gt = F.one_hot(train_label).type(torch.FloatTensor)
    # 定义数据加载器用于训练模型
    train_dataset = VideoDataSet(train_input, train_gt)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def train():

    video_root_path = 'datas'

    # save_val
    save_val = True

    # 读取训练数据
    train_loader = read_data(video_root_path)
    test_loader = train_loader

    # 实例化模型
    model = linear50().cuda()

    # 定义交叉熵损失函数和Adam优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()  # zero the parameter gradients
            # model.Lstm.reset_hidden_state()
            output = model(x.cuda())
            loss = criterion(output, y.cuda())
            loss.backward()  # compute the gradients
            optimizer.step()  # update the parameters with the gradients

            # 输出日志
            if (i+1) % 50 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # val
    if save_val:
        print("pred ...")
        val_file_list = sorted(os.listdir(os.path.join(
            video_root_path, 'test_A', 'test_A_feature')))

        val_pred = {}
        for video in tqdm(val_file_list):
            val_np_data_path = os.path.join(
                video_root_path, 'test_A', 'test_A_feature', video)
            np_data_250 = np.squeeze(np.load(val_np_data_path))

            # 50帧一组数据
            np_5 = []
            for i in range(0, 250, 50):
                np_data_50 = np_data_250[i:i+50]
                np_5.append(np_data_50)

            # 预测
            val_in = torch.from_numpy(np.array(np_5))
            result = model(val_in.cuda())
            val_pred_label = torch.argmax(result, dim=1)
            val_pred_label = val_pred_label.cpu().tolist()

            maxlabel = max(val_pred_label, key=val_pred_label.count)
            val_pred[video] = str(maxlabel)

        # print(val_pred)
        save_test = os.path.join(
            video_root_path, 'test_A_pred_{}.txt'.format(model.name))
        with open(save_test, 'w') as f:
            json.dump(val_pred, f)

        print("save pred file {}".format(save_test))


# train
train()