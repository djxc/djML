
import sys
import torch
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import numpy as np
import xml.etree.ElementTree as ET

class FaceDataset(torch.utils.data.Dataset):
    '''
    '''

    def __init__(self, is_train, imageRoot, train_fold=8):
        self.imageRoot = imageRoot
        self.folds = os.path.join(imageRoot, "FDDB-folds")
        self.imgs_infos = self.__list_files(train_fold)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ]
        )
        print('read ' + str(len(self.imgs)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        img_info = self.imgs_infos[idx]
        bboxes, labels = self.__get_annotation(imgName.replace("jpg", "xml"))
        image = self.__read_image(imgName)
        return image, labels


    def __list_files(self, train_fold):
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
    

    def __parse_ellip_file(self, ellip_path):
        """解析ellip文件"""
        ellip_infos_tmp = []
        with open(ellip_path) as ellip_infos_file:
            ellip_infos = ellip_infos_file.readlines()
        for i, ellip_info in enumerate(ellip_infos):
            if 'big' in ellip_info:
                ellip_info = ellip_info.replace("\n", "")
                if sys.platform =='win32':
                    ellip_info = ellip_info.replace("/", "\\")
                img_path = os.path.join(self.imageRoot, ellip_info + ".jpg")
                ellip_info_tmp = {
                    "image_path": img_path,
                    "coordinates": []
                }
                face_number = int(ellip_infos[i + 1].replace("\n", ""))
                for j in range(face_number):
                    coordinates = ellip_infos[i + 2 + j].split(" ")[:-2]
                    coordinates = [float(c) for c in coordinates]
                    ellip_info_tmp["coordinates"].append(coordinates)
                ellip_infos_tmp.append(ellip_info_tmp)
        return ellip_infos_tmp

    
    def __len__(self):
        return len(self.imgs_infos)
    
    def __get_annotation(self, labelname):
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

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.float32)


    def get_img_info(self, labelname):
        annotation_file = os.path.join(self.imageRoot, "%s" % labelname)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}


    def __read_image(self, img_path):
        image = Image.open(img_path)        
        image = self.transform(image)
        return image.float()
