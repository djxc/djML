# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:55
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : dataset.py
"""

"""
import os
import os.path as osp
import logging
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, img_name)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        assert img.size == mask.size, \
            f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

# one hot数据加载时将每一个类别作为单一通道，设为0或1，然后送入网络运算。


class RSDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, image_file, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        all_image_name = open(image_file, "r")
        self.lines = all_image_name.readlines()
        all_image_name.close()
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

    def __len__(self):
        return len(self.lines)

    def preprocess(self, img_name):
        label1 = cv2.imread(self.masks_dir + "label1/" + img_name)
        label2 = cv2.imread(self.masks_dir + "label2/" + img_name)
        newLabel = label1 * 10 + label2

        
        # 图像旋转,设置随机旋转角度
        angle = random.randint(0, 90)
        rows, cols, channels = newLabel.shape
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
        newLabel = cv2.warpAffine(newLabel, rotate, (cols, rows))
        
        newLabel = newLabel[:, :, 0]    # 读取的是三通道的这里获取其第一个通道数据
        # newLabel = newLabel / 255
        # newLabel = newLabel.transpose((2, 0, 1))
        label_11 = np.where(newLabel == 11, 1, 0)
        label_12 = np.where(newLabel == 12, 1, 0)
        label_13 = np.where(newLabel == 13, 1, 0)
        label_14 = np.where(newLabel == 14, 1, 0)
        label_15 = np.where(newLabel == 15, 1, 0)
        label_16 = np.where(newLabel == 16, 1, 0)

        label_21 = np.where(newLabel == 21, 1, 0)
        label_22 = np.where(newLabel == 22, 1, 0)
        label_23 = np.where(newLabel == 23, 1, 0)
        label_24 = np.where(newLabel == 24, 1, 0)
        label_25 = np.where(newLabel == 25, 1, 0)
        label_26 = np.where(newLabel == 26, 1, 0)

        label_31 = np.where(newLabel == 31, 1, 0)
        label_32 = np.where(newLabel == 32, 1, 0)
        label_33 = np.where(newLabel == 33, 1, 0)
        label_34 = np.where(newLabel == 34, 1, 0)
        label_35 = np.where(newLabel == 35, 1, 0)
        label_36 = np.where(newLabel == 36, 1, 0)

        label_41 = np.where(newLabel == 41, 1, 0)
        label_42 = np.where(newLabel == 42, 1, 0)
        label_43 = np.where(newLabel == 43, 1, 0)
        label_44 = np.where(newLabel == 44, 1, 0)
        label_45 = np.where(newLabel == 45, 1, 0)
        label_46 = np.where(newLabel == 46, 1, 0)

        label_51 = np.where(newLabel == 51, 1, 0)
        label_52 = np.where(newLabel == 52, 1, 0)
        label_53 = np.where(newLabel == 53, 1, 0)
        label_54 = np.where(newLabel == 54, 1, 0)
        label_55 = np.where(newLabel == 55, 1, 0)
        label_56 = np.where(newLabel == 56, 1, 0)

        label_61 = np.where(newLabel == 61, 1, 0)
        label_62 = np.where(newLabel == 62, 1, 0)
        label_63 = np.where(newLabel == 63, 1, 0)
        label_64 = np.where(newLabel == 64, 1, 0)
        label_65 = np.where(newLabel == 65, 1, 0)
        label_66 = np.where(newLabel == 66, 1, 0)

        newLabel = np.array([label_11, label_12, label_13, label_14, label_15, label_16,
                             label_21, label_22, label_23, label_24, label_25, label_26,
                             label_31, label_32, label_33, label_34, label_35, label_36,
                             label_41, label_42, label_43, label_44, label_45, label_46,
                             label_51, label_52, label_53, label_54, label_55, label_56,
                             label_61, label_62, label_63, label_64, label_65, label_66,
                             ])

        img1 = cv2.imread(self.imgs_dir + "im1/" + img_name)
        img2 = cv2.imread(self.imgs_dir + "im2/" + img_name)
        
        rows, cols, channels = img1.shape
        # 图像旋转,设置随机旋转角度
        rotate_im1 = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
        img1 = cv2.warpAffine(img1, rotate_im1, (cols, rows))
        rotate_im2 = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
        img2 = cv2.warpAffine(img2, rotate_im2, (cols, rows))

        img1Bands = cv2.split(img1)
        img2Bands = cv2.split(img2)
        img1Bands.extend(img2Bands)
        newIMG = cv2.merge(img1Bands)
        newIMG = newIMG.transpose((2, 0, 1))

        train_x_transforms = transforms.Compose([          
            transforms.Normalize((0.4914, 0.4822, 0.4465, 0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010, 0.2023, 0.1994, 0.2010))
        ])

        return newIMG, newLabel

    def __getitem__(self, i):
        img_name = self.lines[i].replace("\n", "")
        img, mask = self.preprocess(img_name)
        # print(img.shape)
        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


if __name__ == "__main__":
    dataset = RSDataset("/document/2020/rs_detection/change_detection_train/train/",
                        "/document/2020/rs_detection/change_detection_train/train/",
                        "/document/2020/rs_detection/change_detection_train/train/allImage.txt", 1)
    train_loader = DataLoader(dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    for batch in train_loader:
        batch_imgs = batch['image']
        # batch_masks = batch['mask']
        print(batch_imgs)
