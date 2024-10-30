import os
import random

import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

rootPath = r"D:\Data\spatial_object"
model_root = r"D:\Data\spatial_object"

def split_verfy_train():
    """将数据拆分为验证集以及训练集，比例为1：4
        1、读取每个目标类型下的
    """
    verify_list = []
    train_list = []
    workspace = r"D:\Data\spatial_object\train"
    for i in range(10):
        folder_name = i + 1
        object_folder = os.path.join(workspace, str(folder_name))
        object_list_files = os.path.join(object_folder, "{}.txt".format(folder_name))
        with open(object_list_files) as object_file:
            object_list = object_file.readlines()
            random.shuffle(object_list)
            tmp_verify_list = object_list[:10]
            tmp_train_list = object_list[10:]
            for tmp_verify in tmp_verify_list:
                if tmp_verify[-1] != "\n":
                    tmp_verify = tmp_verify + "\n"
                verify_list.append("{}\\{}".format(object_folder, tmp_verify))
            for tmp_train in tmp_train_list:
                if tmp_train[-1] != "\n":
                    tmp_train = tmp_train + "\n"
                train_list.append("{}\\{}".format(object_folder, tmp_train))
    train_file_path = os.path.join(workspace, "train.txt")
    verify_file_path = os.path.join(workspace, "verify.txt")
    with open(train_file_path, "w") as train_file:
        train_file.writelines(train_list)

    with open(verify_file_path, "w") as verify_file:
        verify_file.writelines(verify_list)

class SpaceObjectDataset(torch.utils.data.Dataset):
    """一个用于加载空间目标分类数据集的自定义数据集。
        1、帆板数类别（0， 1， 2）；载荷数量类别（0， 1）；主体数量类别（0， 1）；个体类别（1-10）
        2、数据结果为[visible_img, SAR_img, 个体one_hot, 主体one_hot, 载荷one_hot, 帆板one_hot]
    """
    def __init__(self, imageFile, mode="train"):
        self.imageDatas = []
        self.mode = mode
        self.cate_ones = torch.sparse.torch.eye(10)
        self.zt_ones = torch.sparse.torch.eye(2)
        self.zh_ones = torch.sparse.torch.eye(2)
        self.fb_ones = torch.sparse.torch.eye(3)

        with open(imageFile) as image_file:
            self.imageDatas = image_file.readlines()

        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),    
                torchvision.transforms.ToTensor(),
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
            image = Image.open(os.path.join(rootPath, imagePath))
            return self.transform(image), imagePath
        else:
            path_info = self.imageDatas[idx].replace("\n", "")
            image_folder = path_info.split(" ")[0]
            object_info = path_info.split("\\")[-1]
            category = int(object_info.split(" ")[2]) - 1
            zt_cate = int(object_info.split(" ")[3])
            zh_cate = int(object_info.split(" ")[4])
            fb_cate = int(object_info.split(" ")[5])

            cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot = self.create_one_hot(category, zt_cate, zh_cate, fb_cate)
            to_tensor = torchvision.transforms.ToTensor()
            visible_imgs = []
            # 分别读取sar和可见光
            for i in range(10):
                tmp_img_path = os.path.join(image_folder, "{}.jpg".format(i + 1))
                image = Image.open(tmp_img_path).convert("L")
                img_tensor = to_tensor(image)
                visible_imgs.append(img_tensor)
            visible_img_plus = torch.cat(visible_imgs, dim=0)

            sar_imgs = []
            for i in range(10, 20):
                tmp_img_path = os.path.join(image_folder, "{}.jpg".format(i + 1))
                image = Image.open(tmp_img_path).convert("L")
                img_tensor = to_tensor(image)
                sar_imgs.append(img_tensor)
            sar_img_plus = torch.cat(sar_imgs, dim=0)            
            return (visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot)

    def __len__(self):
        return len(self.imageDatas)
    
    def create_one_hot(self, category, zt_cate, zh_cate, fb_zate):
        """"""
        cate_one_hot = self.cate_ones.index_select(0, torch.tensor(category))
        zt_one_hot = self.zt_ones.index_select(0, torch.tensor(zt_cate))
        zh_one_hot = self.zh_ones.index_select(0, torch.tensor(zh_cate))
        fb_one_hot = self.fb_ones.index_select(0, torch.tensor(fb_zate))
        return cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot


def load_space_object_data(batch_size):
    ''' 加载数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_path = os.path.join(rootPath, "train", "train.txt")
    train_iter = torch.utils.data.DataLoader(
        SpaceObjectDataset(train_path, "train"), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
    verify_path = os.path.join(rootPath, "train", "verify.txt")
    test_iter = torch.utils.data.DataLoader(
        SpaceObjectDataset(verify_path, "valid"), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter

if __name__ == "__main__":
    split_verfy_train()
    # train_data, verify_data = load_space_object_data(10)
    # for i, (visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot) in enumerate(train_data):  
    #     print(i)      



        
