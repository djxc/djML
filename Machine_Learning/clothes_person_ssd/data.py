
import sys
import cv2
import torch
import random
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import numpy as np
import hashlib
import zipfile, tarfile, requests
import xml.etree.ElementTree as ET

class PersonClothesDataset(torch.utils.data.Dataset):
    '''
    '''

    def __init__(self, is_train, imageRoot):
        self.class_names = ('clothes', 'no_clothes', 'person_clothes', 'person_no_clothes')
        self.imageRoot = imageRoot
        self.imgs = self.list_files()
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ]
        )
        print('read ' + str(len(self.imgs)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        imgName = self.imgs[idx]
        bboxes, labels = self._get_annotation(imgName.replace("jpg", "xml"))
        image = self._read_image(imgName)
        # image, bboxes = random_translate(image, bboxes)
        image, bboxes = random_crop(image, bboxes)
        image, bboxes = random_horizontal_flip(image, bboxes)
        image, bboxes = random_bright(image, bboxes)

        labels = np.c_[labels, bboxes]
        return image, labels

    def list_files(self):
        '''列出目录下所有的jpg文件'''
        fileNames = []
        list_file = os.listdir(self.imageRoot)
        for img in list_file:
            if 'jpg' in img:
                fileNames.append(img)
        return fileNames

    def __len__(self):
        return len(self.imgs)
    
    def _get_annotation(self, labelname):
        annotation_file = os.path.join(self.imageRoot,  "%s" % labelname)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1 / 1920, y1 / 1080, x2 / 1920, y2 / 1080] )
            labels.append([self.class_dict[class_name]])

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.float32)

    def get_img_info(self, labelname):
        annotation_file = os.path.join(self.imageRoot, "%s" % labelname)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, imagename):
        image = Image.open(os.path.join(self.imageRoot, f'{imagename}'))
        # image = torchvision.io.read_image(os.path.join(self.imageRoot, f'{imagename}'))
        # image = image.float()        
        # image = (image- image.mean()) / image.std()
        image = self.transform(image)
        # image_file = os.path.join(self.imageRoot, "%s" % imagename)
        # image = Image.open(image_file).convert("RGB")
        # image = np.array(image)
        return image.float()


def collate_fn(data):
    imgs_list,boxes_list,classes_list=zip(*data)
    assert len(imgs_list)==len(boxes_list)==len(classes_list)
    batch_size=len(boxes_list)
    pad_imgs_list=[]
    pad_boxes_list=[]
    pad_classes_list=[]
 
    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    for i in range(batch_size):
        img=imgs_list[i]
        pad_imgs_list.append(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.))
 
    max_num=0
    for i in range(batch_size):
        n=boxes_list[i].shape[0]
        if n>max_num:max_num=n
    for i in range(batch_size):
        pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
        pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
 
 
    batch_boxes=torch.stack(pad_boxes_list)
    batch_classes=torch.stack(pad_classes_list)
    batch_imgs=torch.stack(pad_imgs_list)
 
    return batch_imgs,batch_boxes,batch_classes


def dataset_collate(batch):
    '''每个batch中数据的shape要完全一致，因此需要将数据全部保持最小的shape或是补充到最大的shape
        collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
        这里图像shape是一致的因此可以不需要修改，而label的shape是不同的因此需要进行修改
    '''
    batch.sort(key=lambda x: len(x[1]), reverse=False)  # 按照数据长度升序排序

    # print(batch[0][1], batch[1][1])
    # print("-------------------------------")
    images = None
    bboxes = None
    max_len = len(batch[len(batch) - 1][1])  # label最长的数据长度 

    for bat in range(0, len(batch)): #
        image = batch[bat][0]
        image = np.expand_dims(image, axis=0)
        # label长度不一致需要进行调整,首先计算与最大长度差，生成长度差数组，追加到原来数据后
        box = batch[bat][1]
        dif_max = max_len - len(box)
        if dif_max > 0:
            dif_maritx = np.zeros((dif_max, 4))
            test = np.ones((dif_max, 1)) * -1
            dif_maritx = np.c_[test,dif_maritx]
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

def load_data_ITCVD(data_root, batch_size):
    ''' 加载ITCVD数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = torch.utils.data.DataLoader(
        PersonClothesDataset(True, data_root), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers
        , collate_fn=dataset_collate)
    # print("load test data")

    test_iter = torch.utils.data.DataLoader(
        PersonClothesDataset(False, "/2020/clothes_person_test/"), batch_size, drop_last=True,
        num_workers=num_workers
        , collate_fn=dataset_collate)
    return train_iter, test_iter




# 图像增强
def random_translate(img, bboxes, p=0.5):
    # 随机平移
    print("translate: ", img.shape)
    if random.random() < p:
        _, h_img, w_img = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]
 
        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
 
        M = np.array([[1, 0, tx], [0, 1, ty]])
        print(M)
        img = cv2.warpAffine(img, M, (w_img, h_img))
 
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes
 
 
def random_crop(img, bboxes, p=0.5):
    # 随机裁剪
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]
 
        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))
 
        img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
 
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes
 
 
# 随机水平反转
def random_horizontal_flip(img, bboxes, p=0.5):
    if random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
    return img, bboxes


# 随机对比度和亮度 (概率：0.5)
def random_bright(img, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        img = img / 255.
    return img, bboxes
 
 
# 随机变换通道
def random_swap(im, bboxes, p=0.5):
    perms = ((0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0))
    if random.random() < p:
        swap = perms[random.randrange(0, len(perms))]
        im[:, :, (0, 1, 2)] = im[:, :, swap]
    return im, bboxes
 
 
# 随机变换饱和度
def random_saturation(im, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        im[:, :, 1] = im[:, :, 1] * random.uniform(lower, upper)
    return im, bboxes
 
 
# 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, bboxes, p=0.5, delta=18.0):
    if random.random() < p:
        im[:, :, 0] = im[:, :, 0] + random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] = im[:, :, 0][im[:, :, 0] > 360.0] - 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] = im[:, :, 0][im[:, :, 0] < 0.0] + 360.0
    return im, bboxes