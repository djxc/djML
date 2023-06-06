import json
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

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