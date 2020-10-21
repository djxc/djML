from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import random
import cv2


def make_dataset(root):
    '''将原数据与mask路径存放在imgs数组中
        @return imgs 保存影像与标注位置对的数组
    '''
    imgs = []
    n = len(os.listdir(root))//2      # 因为原图与mask都在一个文件夹中，所以这里取一半
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))
    return imgs


class DJDataset(Dataset):
    '''根据传入的Dataset(数据源的路径)，加载数据'''

    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''根据index获取对应的图像'''
        x_path, y_path = self.imgs[index]
        # unet定义的图像大小为512*512所以必须输入的图像数据为512*512
        # 自己的数据为1000*1000，因此需要将其切割下
        img_x = cv2.imread(x_path)
        img_x = cv2.resize(img_x, (512, 512))

        img_y = cv2.imread(y_path)
        img_y, a, b = cv2.split(img_y)      # label为三通道，每个通道值一样，所以只取第一通道
        img_y = cv2.resize(img_y, (512, 512))

        # 增加随机旋转，将图像与标注都进行旋转
        angle = random.randint(0, 90)
        img_x = self.rotateIMG(img_x, angle, x_path)
        img_y = self.rotateIMG(img_y, angle, y_path)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        '''返回数据的个数'''
        return len(self.imgs)

    def rotateIMG(self, img, angle, imgName):
        if len(img.shape) == 2:
            rows, cols = img.shape
        elif len(img.shape) == 3:
            rows, cols, _ = img.shape
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), angle, 1)
        newIMG = cv2.warpAffine(img, rotate, (cols, rows))
        return newIMG
