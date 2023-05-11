import os
import torch
import numpy as np


class VideoFeatureDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集。"""
    def __init__(self, file_path, mode="train"):
        self.imageDatas = []
        self.mode = mode
        with open(file_path, encoding="utf-8") as image_file:
            self.imageDatas = image_file.readlines()
       
        self.label_info = ["0", "1", "2", "3", "4"]
        print("read {} {} examples".format(len(self.imageDatas), mode))        

    def __getitem__(self, idx):
        image_path, label, _ = self.imageDatas[idx].replace("\n", "").split(",")
        label = label.strip()
        if self.mode == "test":
            image = self.__open_npy(image_path)
            return (image, image_path)
        elif self.mode in ["train", "verify"]:
            image = self.__open_npy(image_path)
            label_index = self.label_info.index(label)
            label = torch.zeros(1, 5).scatter_(1, torch.tensor([label_index]).unsqueeze(1), 1).squeeze()
            return (image, label)
        
    def __open_npy(self, npy_path):
        depthmap = np.load(npy_path)    #使用numpy载入npy文件
        depthmap = np.squeeze(depthmap, -1)
        depthmap = np.squeeze(depthmap, -1)
        depthmap = torch.from_numpy(depthmap)
        return depthmap

    def __len__(self):
        return len(self.imageDatas)