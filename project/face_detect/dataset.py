
import sys
import torch
from typing import List
from dataclasses import dataclass
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from util import calculate_bbox

@dataclass
class ImageFaceCoordinate:
    image_path: str
    coordinates: List[float]

class FaceDataset(torch.utils.data.Dataset):
    '''人脸数据
        1、
    '''

    def __init__(self, is_train, imageRoot, train_fold=8):
        """
            @param is_train 是训练还是验证
            @param imageRoot 数据根目录
            @param train_fold 训练数据比例，一共10个fold
        """
        self.imageRoot = imageRoot
        self.folds = os.path.join(imageRoot, "FDDB-folds")
        self.imgs_infos = self.__list_files(train_fold)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((300, 400)),
            # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ]
        )
        print('read ' + str(len(self.imgs_infos)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        img_info = self.imgs_infos[idx]
        bboxes_list = []
        for coordinate in img_info.coordinates:
            bboxes = calculate_bbox(coordinate)
            bboxes.insert(0, 1)
            bboxes_list.append(bboxes)
        bboxes = torch.from_numpy(np.array(bboxes_list, dtype=np.float32))
        image = self.__read_image(img_info.image_path)
        return image, bboxes


    def __list_files(self, train_fold) -> List[ImageFaceCoordinate]:
        '''列出目录下所有的fold文件'''
        total_ellip_infos = []
        list_file = os.listdir(self.folds)
        n = 0
        for filename in list_file:
            if filename.endswith("ellipseList.txt") and n <= train_fold:
                file_path = os.path.join(self.folds, filename)
                ellip_infos_tmp = self.__parse_ellip_file(file_path)
                total_ellip_infos.extend(ellip_infos_tmp)
                n = n + 1
        return total_ellip_infos
    

    def __parse_ellip_file(self, ellip_path) -> List[ImageFaceCoordinate]:
        """解析ellip文件
            @param ellip_path 位置文件
            @return 返回图像位置信息列表
        """
        ellip_infos_tmp = []
        with open(ellip_path) as ellip_infos_file:
            ellip_infos = ellip_infos_file.readlines()
        for i, ellip_info in enumerate(ellip_infos):
            if 'big' in ellip_info:
                ellip_info = ellip_info.replace("\n", "")
                if sys.platform =='win32':
                    ellip_info = ellip_info.replace("/", "\\")
                img_path = os.path.join(self.imageRoot, ellip_info + ".jpg")
                ellip_info_tmp = ImageFaceCoordinate(img_path, [])
                face_number = int(ellip_infos[i + 1].replace("\n", ""))
                for j in range(face_number):
                    coordinates = ellip_infos[i + 2 + j].split(" ")[:-2]
                    coordinates = [float(c) for c in coordinates]
                    ellip_info_tmp.coordinates.append(coordinates)
                ellip_infos_tmp.append(ellip_info_tmp)
        return ellip_infos_tmp

    
    def __len__(self):
        return len(self.imgs_infos)



    def __read_image(self, img_path):
        image = Image.open(img_path)     
        if image.layers == 1:
            image = image.convert("RGB")
        image = self.transform(image)
        return image.float()


def dataset_collate_v1(batch):
    """"""
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return imgs, targets

def find_max_width_heigh(batch):
    """找到该批次中图像的最大长宽"""
    max_heigh, max_width = 0, 0
    for b in batch:
        image = b[0]
        _, w, h = image.shape
        if max_heigh < h:
            max_heigh = h
        if max_width < w:
            max_width = w
    return max_heigh, max_width

def dataset_collate(batch):
    '''每个batch中数据的shape要完全一致，因此需要将数据全部保持最小的shape或是补充到最大的shape
        collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
        这里图像shape是不一致，需要遍历图像找到最大长度与宽度将所有图像都设置为填充到最大长和宽，
        label的shape是不同的因此需要进行修改，将所有label尺寸调整为相同
    '''

    batch.sort(key=lambda x: len(x[1]), reverse=False)  # 按照数据长度升序排序
    max_heigh, max_width = find_max_width_heigh(batch)

    images = None
    bboxes = None
    max_len = len(batch[len(batch) - 1][1])  # label最长的数据长度 

    for bat in range(0, len(batch)): #
        box = batch[bat][1]        
        image = batch[bat][0]

        b, w, h = image.shape
        new_image = np.zeros((b, max_width, max_heigh))
        new_image[:, 0:w, 0:h] = image
        image = np.expand_dims(new_image, axis=0)

        # label长度不一致需要进行调整,首先计算与最大长度差，生成长度差数组，追加到原来数据后
        dif_max = max_len - len(box)
        if dif_max > 0:
            dif_maritx = np.zeros((dif_max, 5))
            box = np.append(box, dif_maritx, axis=0)
        box = np.expand_dims(box, axis=0)

        if bboxes is None:
            bboxes = box
            images = image
        else:
            bboxes = np.append(bboxes, box, axis=0)       
            images = np.append(images, image, axis=0)              

    images = torch.tensor(images, dtype=torch.float32)
    boxes = torch.tensor(bboxes, dtype=torch.float32)
    data_copy = (images, boxes)
    return data_copy   