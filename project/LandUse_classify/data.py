
import os
import numpy as np
from PIL import Image
from typing import List
import torch, torchvision
from dataclasses import dataclass

from config import train_dataset_file, verify_dataset_file

LANDUSE_LABELS = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 
    'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 
    'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt']

@dataclass
class MLDataSet:
    train_list: List[str]
    verify_list: List[str]


class LandUseClassifyDataset(torch.utils.data.Dataset):
    """一个用于加载土地利用分类数据集的自定义数据集。"""
    def __init__(self, imageFile, mode="train"):
        self.imageDatas = []
        self.mode = mode
        self.ones = torch.sparse.torch.eye(len(LANDUSE_LABELS))

        with open(imageFile) as image_file:
            self.imageDatas = image_file.readlines()

        self.train_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.RandomVerticalFlip(p=0.5),    
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize([256, 256]),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        print("read {} {} examples".format(len(self.imageDatas), mode))        

    def __getitem__(self, idx):
        if self.mode == "test":
            imagePath = self.imageDatas[idx].replace("\n", "")
            image = Image.open(imagePath)
            return self.transform(image), imagePath
        else:
            imagePath = self.imageDatas[idx].replace("\n", "")
            label = imagePath.split("\\")[-2]
            label = self.ones.index_select(0, torch.tensor(LANDUSE_LABELS.index(label)))
            image = Image.open(imagePath)
            if self.mode == "train":
                image = self.train_transform(image)
            else:
                image = self.transform(image)
            return (image, label)

    def __len__(self):
        return len(self.imageDatas)


def split_train_verify_list(data_dir) -> MLDataSet:
    """拆分为测试集与验证集"""
    folder_list = os.listdir(data_dir)
    train_list = []
    verify_list = []
    for folder in folder_list:
        if os.path.isdir(os.path.join(data_dir, folder)):
            ml_dataset = split_train_verify(os.path.join(data_dir, folder))
            train_list.extend(ml_dataset.train_list)
            verify_list.extend(ml_dataset.verify_list)
    return MLDataSet(train_list, verify_list)


def split_train_verify(data_dir, ratio = 0.7) -> MLDataSet:
    """将当前文件夹下的数据拆分为训练集与验证集"""
    file_list = os.listdir(data_dir)
    file_num = len(file_list)
    train_list = []
    verify_list = []
    np.random.shuffle(file_list)
    train_num = file_num * ratio
    for i, file in enumerate(file_list):
        file_path = os.path.join(data_dir, file)
        if i < train_num:
            train_list.append(file_path)
        else:
            verify_list.append(file_path)
    return MLDataSet(train_list, verify_list)
    
    
def save_train_verify_info():
    ml_dataset = split_train_verify_list(r"E:\Data\MLData\classify\UCMerced_LandUse\Images")
    with open(r"E:\Data\MLData\classify\UCMerced_LandUse\train.txt", "w") as train_file:
        np.random.shuffle(ml_dataset.train_list)
        for file in ml_dataset.train_list:
            train_file.write(file + "\n")
    with open(r"E:\Data\MLData\classify\UCMerced_LandUse\verify.txt", "w") as verify_file:
        np.random.shuffle(ml_dataset.verify_list)
        for file in ml_dataset.verify_list:
            verify_file.write(file + "\n")

def load_land_use_dataset(batch_size, num_workers = 4):
    """加载数据，训练数据以及测试数据"""
    train_landUse_dataset = LandUseClassifyDataset(train_dataset_file)
    train_iter = torch.utils.data.DataLoader(train_landUse_dataset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    verify_landUse_dataset = LandUseClassifyDataset(verify_dataset_file, "verify")
    verify_iter = torch.utils.data.DataLoader(verify_landUse_dataset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return train_iter, verify_iter

if __name__ == "__main__":
    # save_train_verify_info()
    batch_size = 4
    num_workers = 4
    landUse_dataset = LandUseClassifyDataset(r"E:\Data\MLData\classify\UCMerced_LandUse\train.txt")

    train_iter = torch.utils.data.DataLoader(landUse_dataset, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    for i, (features, labels) in enumerate(train_iter):
        print(len(features), labels)    
