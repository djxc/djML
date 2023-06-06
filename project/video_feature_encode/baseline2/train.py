#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from baseline2model import linear50
from baseline2Data import read_data

# 设置超参数
num_classes = 5
batch_size = 32
learning_rate = 0.001
num_epochs = 50
video_root_path = 'datas'



def train():
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
       

def test():    
    print("pred ...")
    model = linear50().cuda()
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


if __name__ == "__main__":
    train()