import random
import torch
import torchvision
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms.functional



class RoadDataset(Dataset):
    '''根据传入的Dataset(数据源的路径)，加载数据'''

    def __init__(self, file_path, train_mode='train', transform=None, target_transform=None):
        self.train_mode = train_mode
        self.imgs, self.labels = self.read_img_file(file_path)        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''根据index获取对应的图像'''
        if self.train_mode == "train" or self.train_mode == "verify":
            x_path = self.imgs[index]
            y_path = self.labels[index]
            img_x = Image.open(x_path)
            img_y = Image.open(y_path)
            if self.train_mode == 'train':
                # 在训练阶段增加随机旋转，将图像与标注都进行旋转
                angle = random.randint(0, 90)
                # img_x = self.rotateIMG(img_x, angle)
                # img_y = self.rotateIMG(img_y, angle)

        else:
            x_path = self.imgs[index]
            img_x = Image.open(x_path)
            img_y = np.zeros([512, 512])

        if self.transform is not None:
            img_x = self.transform(img_x)

        img_y = np.array(img_y)
        img_y = np.where(img_y == 255, 1, 0)
        img_y = img_y.astype(np.float32)
        img_y = torch.tensor(img_y).unsqueeze(0)
        return img_x, img_y, x_path

    def __len__(self):
        '''返回数据的个数'''
        return len(self.imgs)
    
    def read_img_file(self, file_path: str):
        """从文件中读取图像与lebel位置"""
        img_paths = []
        label_paths = []
        if self.train_mode == "train" or self.train_mode == "verify":            
            with open(file_path) as train_img_label_file:
                train_img_labels = train_img_label_file.readlines()        
                for img_label in train_img_labels:
                    img_and_label = img_label.replace("\n", "").split(",")
                    img_paths.append(img_and_label[0])
                    label_paths.append(img_and_label[1])
        else:
            with open(file_path) as test_img_file:
                test_imgs = test_img_file.readlines()
                for test_img in test_imgs:
                    img_paths.append(test_img.replace("\n", ""))

        return img_paths, label_paths


    def rotateIMG(self, img, angle):
        return torchvision.transforms.functional.rotate(img, angle)
        
