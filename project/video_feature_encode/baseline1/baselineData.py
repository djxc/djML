import os
import os.path
import numpy as np
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset


from baseline1Utils import noise_injection


class VideoDataset(Dataset):
    def __init__(self, data_path, label_path=None, train="train", transform=None, val_ratio=0.2):
        self.data_path = data_path
        self.transform = transform
        self.train = train
        
        self.file_list = sorted(os.listdir(self.data_path))
        n_samples = len(self.file_list)
        split_idx = int(n_samples * val_ratio)
        if train:
            self.file_list = self.file_list[split_idx:]
            with open(label_path, 'r') as f:
                labels = eval(f.read())
                self.labels = labels
        else:
            if label_path is not None:
                self.file_list = self.file_list[:split_idx]
                with open(label_path, 'r') as f:
                    labels = eval(f.read())
                self.labels = labels
            else:
                self.file_list = self.file_list
                self.labels = None


    def __getitem__(self, index):
        data = torch.from_numpy(np.load(os.path.join(self.data_path, self.file_list[index]))).squeeze()
        if self.labels is not None:
            label = int(self.labels[self.file_list[index]])
        else:
            label = -1  # use dummy label for test set
        
        if self.transform:
            # 噪音为0.1对结果影响不大
            data = noise_injection(data, noise_factor=0.2)
        
        return data, label

    def __len__(self):
        return len(self.file_list)
    

class VideoDataset1(Dataset):
    def __init__(self, label_path=None, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.transform_f = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 2048))
        ])
        with open(label_path, encoding="utf-8") as image_file:
            self.imageDatas = image_file.readlines()
        

    def __getitem__(self, idx):
        if self.train in ["train", "verify"]:
            image_path, label, _ = self.imageDatas[idx].replace("\n", "").split(",")
            label = int(label.strip())
        else:
            image_path = self.imageDatas[idx].replace(",\n", "")
            label = image_path
        # data = np.load(image_path)    #使用numpy载入npy文件
        # data = np.squeeze(data, -1)
        # data = np.squeeze(data, -1)
        data = torch.from_numpy(np.load(image_path)).squeeze()  # 250*2048
        if self.transform:
            # data = self.random_remove_frame_item(data)
            # data = self.transform_f(data)
            # 噪音为0.1对结果影响不大
            nosize_factor = random.randint(0, 5) * 0.1
            if nosize_factor > 0 and nosize_factor <= 0.2:
                data = noise_injection(data, noise_factor=nosize_factor)
        return data, label

    def __len__(self):
        return len(self.imageDatas)
    
    def random_remove_frame_item(self, image_data):
            """随机去除部分数据,最大去除为三分之一
                1、图像高度为250乘去除比例随机生产0-0.3
                2、
            """
            start_line = random.randint(0, 10)
            step_num = random.randint(1, 5)
            new_image = image_data[start_line::step_num]
            return new_image

def random_remove_frame(image_data_batch):
        """随机去除部分数据,最大去除为三分之一
            1、图像高度为250乘去除比例随机生产0-0.3
            2、
        """
        batch_size, h, w = image_data_batch.shape
        new_image = torch.zeros((batch_size, 50, w))
        start_line = random.randint(0, 4)
        for i, image_data in enumerate(image_data_batch):
            new_image[i] = image_data[start_line::5]
        return new_image

def split_frame(image_data_batch):
        """预测时将数据拆分为5分，模型对五分数据进行预测
        """
        new_image = torch.zeros((5, 50, 2048))        
        for i in range(5):
            new_image[i] = image_data_batch[0][i::5]
        return new_image