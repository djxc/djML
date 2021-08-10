
import sys
import torch
import torchvision
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
        print('read ' + str(len(self.imgs)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        imgName = self.imgs[idx]
        labels = self._get_annotation(imgName.replace("jpg", "xml"))
        image = self._read_image(imgName)
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
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([self.class_dict[class_name], x1 / 1920, y1 / 1080, x2 / 1920, y2 / 1080] )

        return np.array(boxes, dtype=np.float32)

    def get_img_info(self, labelname):
        annotation_file = os.path.join(self.imageRoot, "%s" % labelname)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, imagename):
        image = torchvision.io.read_image(os.path.join(self.imageRoot, f'{imagename}'))
        image = image.float()
        image = (image- image.mean()) / image.std()
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
    images = []
    bboxes = None
    for img, box in batch:
        images.append(img)
        box = np.expand_dims(box, axis=0)
        print("box: ", box.shape)
        if bboxes is None:
            bboxes = box
        else:
            bboxes = np.append(bboxes, box, axis=0)       
            print("bboxes: ", bboxes.shape)
    images = np.array(images)
    bboxes = np.array(bboxes)
    print(images.shape, bboxes.shape)
    return images, bboxes

def load_data_ITCVD(data_root, batch_size):
    ''' 加载ITCVD数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = torch.utils.data.DataLoader(
        PersonClothesDataset(True, data_root), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
        #, collate_fn=dataset_collate)
    # print("load test data")

    test_iter = torch.utils.data.DataLoader(
        PersonClothesDataset(False, "/2020/clothes_person_test/"), batch_size, drop_last=True,
        num_workers=num_workers)
        #, collate_fn=dataset_collate)
    return train_iter, test_iter