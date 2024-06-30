from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import random
import cv2



class RoadDataset(Dataset):
    '''根据传入的Dataset(数据源的路径)，加载数据'''

    def __init__(self, train_file, verify_file, trainOrTest='train', transform=None, target_transform=None):
        self.train_imgs_labels, self.verify_imgs_labels = self.read_img_file(train_file, verify_file)        
        self.trainOrTest = trainOrTest
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''根据index获取对应的图像'''
        x_path, y_path = self.train_imgs_labels[index].replace("\n", "").split(",")
        # unet定义的图像大小为512*512所以必须输入的图像数据为512*512
        # 自己的数据为1000*1000，因此需要将其切割下
        img_x = Image.open(x_path)

        img_y = Image.open(y_path)

        # if self.trainOrTest == 'train':
        #     # 在训练阶段增加随机旋转，将图像与标注都进行旋转
        #     angle = random.randint(0, 90)
        #     img_x = self.rotateIMG(img_x, angle, x_path)
        #     img_y = self.rotateIMG(img_y, angle, y_path)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        '''返回数据的个数'''
        return len(self.train_imgs_labels)
    
    def read_img_file(self, train_file: str, verify_file: str):
        """从文件中读取图像与lebel位置"""
        with open(train_file) as train_img_label_file:
            train_img_labels = train_img_label_file.readlines()

        with open(verify_file) as verify_img_label_file:
            verify_img_labels = verify_img_label_file.readlines()

        return train_img_labels, verify_img_labels


    def rotateIMG(self, img, angle, imgName):
        if len(img.shape) == 2:
            rows, cols = img.shape
        elif len(img.shape) == 3:
            rows, cols, _ = img.shape
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
        newIMG = cv2.warpAffine(img, rotate, (cols, rows))
        return newIMG
