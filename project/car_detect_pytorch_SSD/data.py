

import os
import torch
import random
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET

from config import workspace, car_class_list

def split_data():
    """将数据拆分为训练集和验证集8:2
        1、首先将数据打乱，拆分为10份
        2、选取其中2份为验证集，其他的为训练集；然后再选择其他两份为验证集，其他为训练集，依次循环5次
    """
    img_folder = os.path.join(workspace, "input_path")
    label_folder = os.path.join(workspace, "gt")
    img_list = os.listdir(img_folder)
    random.shuffle(img_list)

    ten_size = int(len(img_list) / 10)
    for i in range(5):
        verify_img_list = img_list[i * ten_size * 2 : (i + 1) * ten_size * 2]
        train_img_list = img_list[:i * ten_size * 2]
        train_img_list2 = img_list[(i + 1) * ten_size * 2:]
        train_img_list.extend(train_img_list2)
        with open(os.path.join(workspace, "train_{}.csv".format(i + 1)), "w") as img_path_file:
            for train_img in train_img_list:
                img_path = os.path.join(img_folder, train_img)
                label_path = os.path.join(label_folder, train_img.replace(".tif", ".xml"))
                img_path_file.write("{},{}\n".format(img_path, label_path))

        with open(os.path.join(workspace, "verify_{}.csv".format(i + 1)), "w") as img_path_file:
            for verify_img in verify_img_list:
                img_path = os.path.join(img_folder, verify_img)
                label_path = os.path.join(label_folder, train_img.replace(".tif", ".xml"))
                img_path_file.write("{},{}\n".format(img_path, label_path))

class CarDataset(torch.utils.data.Dataset):
    """一个用于加载汽车检测数据集的自定义数据集。"""
    def __init__(self, is_train, workspace, fold_num=1):
        self.is_train = is_train
        self.fold_num = fold_num
        self.image_paths, self.label_paths = self.read_data_car(workspace)     
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(10000),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )       
        print('read ' + str(len(self.image_paths)) + (
            f' training examples' if is_train else f' validation examples'))

    def read_data_car(self, workspace: str):
        """读取汽车检测数据集中的图像和标签路径。"""
        image_paths = []
        label_paths = []
        csv_fname = os.path.join(workspace, 'train_{}.csv'.format(self.fold_num) if self.is_train else 'verify_{}.csv'.format(self.fold_num))
        with open(csv_fname) as csv_file:
            car_data_list = csv_file.readlines()
            for car_data in car_data_list:
                img_label = car_data.replace("\n", "").split(",")
                image_paths.append(img_label[0])                
                label_paths.append(img_label[1])
        return image_paths, label_paths
        # csv_data = pd.read_csv(csv_fname)
        # csv_data = csv_data.set_index('img_name')
        # images, targets = [], []
        # for img_name, target in csv_data.iterrows():
        #     images.append(
        #         torchvision.io.read_image(
        #             os.path.join(data_dir,
        #                         'bananas_train' if is_train else 'bananas_val',
        #                         'images', f'{img_name}')))
        #     # Here `target` contains (class, upper-left x, upper-left y,
        #     # lower-right x, lower-right y), where all the images have the same
        #     # banana class (index 0)
        #     targets.append(list(target))
        # return images, torch.tensor(targets).unsqueeze(1) / 256

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        feature = Image.open(img_path)
        feature = feature.convert('RGB')

        label_info = ET.parse(label_path)  # 替换成你的XML文件路径
        root = label_info.getroot()
        size_info = root.find('size')
        width_size = int(size_info.find("width").text)
        height_size = int(size_info.find("height").text)
        object_list = root.find('objects').findall("object")
        label_list = []
        for object in object_list:
            label_tmp = []
            car_class = object.find("possibleresult").find("name").text 
            label_tmp.append(car_class_list.index(car_class))
            points = object.find("points")
            for i, point in enumerate(points):
                if i == 0 or i == 3:
                    point_txt = point.text.split(",")
                    width = float(point_txt[0])/width_size
                    height = float(point_txt[1])/height_size
                    label_tmp.append(width)
                    label_tmp.append(height)
            label_list.append(label_tmp)
        return (self.transform(feature), torch.tensor(label_list))

    def __len__(self):
        return len(self.image_paths)
    
if __name__ == "__main__":
    split_data()
