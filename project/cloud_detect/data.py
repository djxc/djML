
import sys
import math
import torch
import random
from pathlib import Path
import torchvision
from osgeo import gdal
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import numpy as np
import xml.etree.ElementTree as ET

class CloudDataset(torch.utils.data.Dataset):
    '''
        云检测数据集
    '''

    def __init__(self, is_train, root_folder):
        self.meta_name = "38-Cloud_Training_Metadata_Files"
        self.root_folder = root_folder
        self.train_folder = os.path.join(root_folder, "38-Cloud_training")
        self.aux_folder = os.path.join(root_folder, self.meta_name)
        self.aux_infos = self.__get_aux_info()
        self.imgs_infos = self.__get_image_list()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
            ]
        )
        print('read ' + str(len(self.imgs_infos)) + (f' training examples' if is_train else f' validation examples'))

    def __get_aux_info(self):
        """获取元数据信息
            band2:blue
            band3:green
            band4:red
            band5:nir
        """
        aux_files = os.listdir(os.path.join(self.aux_folder, self.meta_name))
        aux_infos = {}
        for aux_file in aux_files:
            aux_path = os.path.join(self.aux_folder, self.meta_name, aux_file)
            aux_info = {}
            with open(aux_path) as aux_info_file:
                aux_content = aux_info_file.readlines()
                for aux in aux_content:
                    if "SUN_ELEVATION" in aux:
                        aux_info["sun_ele"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_MULT_BAND_2" in aux:
                        aux_info["ref_b2"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_MULT_BAND_3" in aux:
                        aux_info["ref_b3"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_MULT_BAND_4" in aux:
                        aux_info["ref_b4"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_MULT_BAND_5" in aux:
                        aux_info["ref_b5"] = float(aux.replace("\n", "").split(" = ")[-1])

                    elif "REFLECTANCE_ADD_BAND_2" in aux:
                        aux_info["ref_add_b2"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_ADD_BAND_3" in aux:
                        aux_info["ref_add_b3"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_ADD_BAND_4" in aux:
                        aux_info["ref_add_b4"] = float(aux.replace("\n", "").split(" = ")[-1])
                    elif "REFLECTANCE_ADD_BAND_5" in aux:
                        aux_info["ref_add_b5"] = float(aux.replace("\n", "").split(" = ")[-1])
            aux_name = Path(aux_path).stem
            aux_infos[aux_name] = aux_info
        return aux_infos

    def __getitem__(self, idx):
        img_path = self.imgs_infos[idx]
        image, label = self.__read_image(img_path)
        return image, label


    def __get_image_list(self):
        '''列出训练集所有影像数据'''
        imgs_list_file = os.path.join(self.train_folder, "training_patches_38-cloud_nonempty.csv")
        with open(imgs_list_file) as imgs_files:
            imgs_list = imgs_files.readlines()
        imgs_list = imgs_list[1:]
        random.shuffle(imgs_list)
        # 先选一部分数据进行测试
        imgs_list = imgs_list[:400]
        return imgs_list

    
    def __len__(self):
        return len(self.imgs_infos)    

    def __read_image(self, img_path):
        """读取图像，分别读取BGRN以及gt"""
        img_path = img_path.replace("\n", "") + ".TIF"
        gt_path = os.path.join(self.train_folder, "train_gt", "gt_{}".format(img_path))
        b_path = os.path.join(self.train_folder, "train_blue", "blue_{}".format(img_path))
        g_path = os.path.join(self.train_folder, "train_green", "green_{}".format(img_path))
        r_path = os.path.join(self.train_folder, "train_red", "red_{}".format(img_path))
        n_path = os.path.join(self.train_folder, "train_nir", "nir_{}".format(img_path))

        # 云255，其他为0
        gt_img = self.__get_array_from_dataset(gt_path, isGt=True)
        gt_img = np.expand_dims(gt_img, axis=0)
        # 将每个波段转换为反射率，读取该文件对应的元数据。最后将四个波段合并在一起
        # 表层反射率 = (DN * Mp + Ap) / sin(太阳高度角)
        b_img = self.__get_array_from_dataset(b_path, isGt=True, band_num=2) 
        g_img = self.__get_array_from_dataset(g_path, isGt=True, band_num=3) 
        r_img = self.__get_array_from_dataset(r_path, isGt=True, band_num=4) 
        n_img = self.__get_array_from_dataset(n_path, isGt=True, band_num=5) 
        img = np.stack((b_img, g_img, r_img, n_img))
        # image = self.transform(image)
        return img, gt_img
    
    def __get_array_from_dataset(self, img_path, isGt=False, band_num = 2):
        """"""
        dataset = gdal.Open(img_path)
        img = dataset.GetRasterBand(1).ReadAsArray()
        if isGt:
            img = np.where(img > 0, 1, 0).astype(np.float32)
        else:
            img_name = "LC08" + img_path.split("LC08")[-1]
            aux_info = self.aux_infos[img_name]
            img = (img * aux_info["ref_b" + band_num] + aux_info["ref_add_b" + band_num]) / math.sin(aux_info["sun_ele"])
            img = img.astype(np.float32)

        return img
