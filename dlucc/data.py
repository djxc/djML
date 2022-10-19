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
    def __init__(self, imageFile, mode="train"):
        self.imageDatas = []
        self.mode = mode
        self.ones = torch.sparse.torch.eye(len(categories))
        with open(imageFile) as image_file:
            self.imageDatas = image_file.readlines()

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
            imagePath, label = self.imageDatas[idx].replace("\n", "").split(", ")        
            label = self.ones.index_select(0, torch.tensor(categories.index(label))) # categories.index(label)
            image = Image.open(os.path.join(rootPath, imagePath))
            image = self.transform(image)
            return (image, label)

    def __len__(self):
        return len(self.imageDatas)

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
    data_analysis()
