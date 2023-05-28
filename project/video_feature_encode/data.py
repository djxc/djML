import os
import torch
import random
import numpy as np
from torchvision import transforms

IMG_HEIGH = 250
IMG_WIDTH = 2048

class VideoFeatureDataset(torch.utils.data.Dataset):
    """用于加载视觉特征数据集"""
    def __init__(self, file_path, mode="train"):
        self.imageDatas = []
        self.mode = mode
        self.use_cuda = False
        if torch.cuda.is_available():
            self.use_cuda = True
        with open(file_path, encoding="utf-8") as image_file:
            self.imageDatas = image_file.readlines()
        if mode == "train":
            self.transform_norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((IMG_HEIGH, IMG_WIDTH))
            ])
        else:
            self.transform_norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

        self.label_info = ["0", "1", "2", "3", "4"]
        print("read {} {} examples".format(len(self.imageDatas), mode))        

    def __getitem__(self, idx):
        if self.mode == "test":
            image_path = self.imageDatas[idx].replace(",\n", "")
            image = self.__open_npy(image_path)
            return (image, image_path)
        elif self.mode in ["train", "verify"]:
            image_path, label, _ = self.imageDatas[idx].replace("\n", "").split(",")
            label = label.strip()
            image = self.__open_npy(image_path)
            label_index = self.label_info.index(label)
            label = torch.zeros(1, 5).scatter_(1, torch.tensor([label_index]).unsqueeze(1), 1).squeeze()
            return (image, label)
        
    def __open_npy(self, npy_path):
        depthmap = np.load(npy_path)    #使用numpy载入npy文件
        depthmap = np.squeeze(depthmap, -1)
        depthmap = np.squeeze(depthmap, -1)
        # 随机去除帧
        if self.mode == "train":
            depthmap = self.__random_remove_frame(depthmap)
        depthmap = self.transform_norm(depthmap)
        depthmap = depthmap.float()
        return depthmap

    def __random_remove_frame(self, image_data):
        """随机去除部分数据,最大去除为三分之一
            1、图像高度为250乘去除比例随机生产0-0.3
            2、

        """
        remove_radio = random.randint(0, 30)/100
        if remove_radio == 0.0 :
            return image_data
        remove_count = int(IMG_HEIGH * remove_radio)
        remove_step = int(IMG_HEIGH / remove_count)
        result_total_line = IMG_HEIGH - remove_count 
        result = np.zeros((result_total_line, IMG_WIDTH))
        result_count = 0
        for i in range(0, IMG_HEIGH, remove_step):
            line_count = remove_step - 1
            end_line = i + line_count
            if end_line > IMG_HEIGH:
                line_count = IMG_HEIGH - i - 1
                end_line = IMG_HEIGH
            result_end = result_count + line_count
            if result_count + line_count > result_total_line:
                result_end = result_total_line
            if end_line - i > result_end - result_count:
                end_line = i + (result_end - result_count)
            result[result_count : result_end] = image_data[i:end_line]
            result_count = result_end
        return result


    def __len__(self):
        return len(self.imageDatas)