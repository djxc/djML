import os, shutil
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


rootPath = r"D:\Data\MLData\segment\LUCC"

class LUCCDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集。"""
    def __init__(self, file_name, mode="train", transform = None, label_transform = None):
        self.imageDatas = []
        self.mode = mode
        with open(os.path.join(rootPath, file_name)) as image_file:
            self.imageDatas = image_file.readlines()

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.label_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()               
            ]
        )
        print("read {} {} examples".format(len(self.imageDatas), mode))        

    def __getitem__(self, idx):
        image_name = self.imageDatas[idx].replace("\n", "").replace(",", "")
        if self.mode == "test":
            imagePath = self.imageDatas[idx].replace("\n", "")
            image = Image.open(os.path.join(rootPath, imagePath))
            return self.transform(image), imagePath
        else:
            imagePath = os.path.join(rootPath, "train_data", "img_train", image_name)
            labelPath = os.path.join(rootPath, "train_data", "lab_train", image_name.replace(".jpg", ".png"))
            image = Image.open(imagePath)
            label = Image.open(labelPath)
            image = self.transform(image)
            label = self.label_transform(label)
            return (image, label)

    def __len__(self):
        return len(self.imageDatas)

def splitTrainAndVerify():
    '''将数据分为训练集以及验证集'''
    files = os.listdir(os.path.join(rootPath, "train_data", "img_train"))
    imageNum = len(files)
    np.random.shuffle(files)
    np.random.shuffle(files)    
    verify_list = ""
    train_list = ""
    for i, image in enumerate(files):
        if i <= imageNum * 0.2:
            verify_list = verify_list + image + ",\n"
        else:
            train_list = train_list + image + ",\n"

    with open(os.path.join(rootPath, "train.csv"), "w+") as train_file:
        train_file.write(train_list)
    
    with open(os.path.join(rootPath, "verify.csv"), "w+") as verify_file:
        verify_file.write(verify_list)


def data_analysis():
    """分析当前数据情况"""
    lab_path = os.path.join(rootPath, "train_data", "lab_train")
    img_path_list = os.listdir(lab_path)
    value_pos = [0, 0, 0, 0, 0, 0, 0]
    for img_path in tqdm(img_path_list):
        image = Image.open(os.path.join(lab_path, img_path))
        image = np.array(image)
        value = set(image.flatten().tolist())
        for v in value:
            if v < 255:
                value_pos[v] = value_pos[v] + 1       
    print(value_pos)


if __name__ == "__main__":
    # data_analysis()
    splitTrainAndVerify()
