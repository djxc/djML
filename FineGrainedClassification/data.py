# 读取数据类
# #
import os
import torch
import random
import torchvision
import numpy as np
import imageio
import cv2
from PIL import Image

class UCMLanduseDataset(torch.utils.data.Dataset):
    """加载UCMLanduse数据
        1、首先获取获取数据目录下的每个文件夹即为所有的类
        2、然后将数据集划分为训练集以及验证集可以按照7：3规则划分
        3、根据当前是训练还是验证进行划分
    """
    def __init__(self, dataFile, crop_size):
        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.Resize((256, 256))
        ])
        self.crop_size = crop_size
        imageFile = open(dataFile, "r")
        self.imagesLabels = imageFile.readlines()

    def __getitem__(self, idx):
        label, imagePath = self.imagesLabels[idx].split(", ")
        imagePath = imagePath.replace("\n", "")
        image = Image.open(imagePath).convert('RGB')
        image = np.array(image, dtype="float32")
        image = torch.from_numpy(image).permute(2, 0, 1) 
        image = self.transforms(image)
        # print(label, image.shape)
        # image = torchvision.io.read_image(imagePath.replace("\n", ""))
        label = torch.tensor(float(label))
        return label, image

    def __len__(self):
        return len(self.imagesLabels)


def splitUCMLanduseData(root_path):
    '''将UCMLanduse数据集进行划分
        1、获取类别
        2、进入每个类别子文件夹，然后按照7：3将其划分为训练集与验证集
        3、最后写入文件内
    '''
    landuseCls = os.listdir(root_path)
    train_data = []
    val_data = []
    for landuse in landuseCls:
        imagesClsPath = os.path.join(root_path, landuse)
        images = os.listdir(imagesClsPath)
        random.shuffle(images)
        random.shuffle(images)
        train_images = images[0:70]
        val_images = images[70:len(images)]
        landuseClsIndex = str(landuseCls.index(landuse))
        for trainImage in train_images:
            train_data.append(landuseClsIndex + ", " + os.path.join(imagesClsPath, trainImage) + "\n")
        for valImage in val_images:
            val_data.append(landuseClsIndex + ", " + os.path.join(imagesClsPath, valImage) + "\n")
        print(landuse)
    random.shuffle(train_data)
    random.shuffle(train_data)
    random.shuffle(val_data)
    with open(os.path.join(root_path, "train_data.txt"), 'w') as trainFile:
        trainFile.writelines(train_data)
    with open(os.path.join(root_path, "val_data.txt"), 'w') as valFile:
        valFile.writelines(val_data)
    print(len(train_data), len(val_data))

# 加载细粒度分类数据集cub200-211#
class CUB():
    def __init__(self, root, is_train=True, data_len=None,transform=None, target_transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        # 图片索引
        img_name_list = []
        for line in img_txt_file:
            # 最后一个字符为换行符
            img_name_list.append(line[:-1].split(' ')[-1])

        # 标签索引，每个对应的标签减１，标签值从0开始
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)

        # 设置训练集和测试集
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # zip压缩合并，将数据与标签(训练集还是测试集)对应压缩
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
        # 然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
        # 我们可以使用 list() 转换来输出列表

        # 如果 i 为 1，那么设为训练集
        # １为训练集，０为测试集
        # zip压缩合并，将数据与标签(训练集还是测试集)对应压缩
        # 如果 i 为 1，那么设为训练集
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        train_label_list = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        test_label_list = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
        if self.is_train:
            # scipy.misc.imread 图片读取出来为array类型，即numpy类型
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            # 读取训练集标签
            self.train_label = train_label_list
        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list

    # 数据增强
    def __getitem__(self,index):
        # 训练集
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
        # 测试集
        else:
            img, target = self.test_img[index], self.test_label[index]

        if len(img.shape) == 2:
            # 灰度图像转为三通道
            img = np.stack([img]*3,2)
        # 转为 RGB 类型
        img = Image.fromarray(img,mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)