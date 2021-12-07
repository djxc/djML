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

root_path = "/2020/data/landuse"
landuse_class = 21
usePhone_path = "/2020/data/usePhone/train/"
usePhone_path_test = "/2020/data/usePhone/test_images_a/"


class UCMLanduseDataset(torch.utils.data.Dataset):
    """加载UCMLanduse数据
        1、首先获取获取数据目录下的每个文件夹即为所有的类
        2、然后将数据集划分为训练集以及验证集可以按照7：3规则划分
        3、根据当前是训练还是验证进行划分
    """
    def __init__(self, dataFile, crop_size):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.crop_size = crop_size
        imageFile = open(dataFile, "r")
        self.imagesLabels = imageFile.readlines()

    def __getitem__(self, idx):
        label, imagePath = self.imagesLabels[idx].split(", ")
        # 这里不需要将label转换为one-hot编码，在计算交叉熵损失函数时会自动转换为one-hot编码
        # label_ = np.zeros(landuse_class)
        # label_[int(label)] = 1
        imagePath = imagePath.replace("\n", "").replace("\\", "/")
        imagePath = root_path + imagePath
        image = Image.open(imagePath)
        image = self.transforms(image)
        # label_ = torch.from_numpy(label_)
        label = torch.tensor(float(label))
        return label.long(), image

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

class UsePhoneDataset(torch.utils.data.Dataset):
    """加载usePhone数据
        1、首先获取获取数据目录下的每个文件夹即为所有的类
        2、然后将数据集划分为训练集以及验证集可以按照7：3规则划分
        3、根据当前是训练还是验证进行划分
    """
    def __init__(self, dataFile, isTrain=True, testModel=False):
        # 训练模式下进行数据增强，验证与测试模式下不进行增强处理
        if isTrain:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.5),     # 随机改变亮度
                # 随机改变图像的色调
                torchvision.transforms.ColorJitter(hue=0.5),
                # 随机改变图像的对比度
                torchvision.transforms.ColorJitter(contrast=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = torchvision.transforms.Compose([                
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.testModel = testModel
        self.isTrain = isTrain
        imageFile = open(dataFile, "r")
        self.imagesLabels = imageFile.readlines()

    def __getitem__(self, idx):
        # 如果测试则不进行数据增强，label返回的是文件名称;如果为验证则不进行数据增强，label返回真实的label
        if not self.isTrain:
            imagePath = self.imagesLabels[idx]
            if self.testModel:
                imagePath = imagePath.replace("\n", "")
                label = imagePath
                imagePath = usePhone_path_test + imagePath
            else:
                label, imagePath = self.imagesLabels[idx].split(",")
                imagePath = imagePath.replace("\n", "")
                label = torch.tensor(float(label)).long()
                imagePath = usePhone_path + imagePath
            image = Image.open(imagePath)
        else:
            label, imagePath = self.imagesLabels[idx].split(",")
            # 这里不需要将label转换为one-hot编码，在计算交叉熵损失函数时会自动转换为one-hot编码
            imagePath = imagePath.replace("\n", "")
            imagePath = usePhone_path + imagePath            
            image = Image.open(imagePath)
    
            # 如果为存在手机，则将目标部分替换为其他数据，然后将该label修改为1
            # 图像中存在手机需要在目标范围内进行随机移动
            if label == "0":
                pass
                # randNum = random.randint(0, 20)

                #if randNum > 19:
                #    bboxPath = imagePath.replace("JPEGImages", "labels").replace(".jpg", ".txt")
                #    image = self.removeLabel(image, bboxPath)
                #    label = "1"
            # 如果不存在手机，则随机读取包含手机图像，将目标区域随机替换到该数据上，然后将该label修改为0
            # 如果不存在手机可以进行平移处理label不变
            else:
                randNum = random.randint(0, 9)
                if randNum > 6:
                    image, isChnage, lam = self.addLabel(image)
                    if isChnage:
                        label = lam

            # 随机进行放大缩小，放大缩小为等比例
            sizeRange = [0.5, 0.7, 1.2, 1.3]
            width, height = image.size
            randSize = random.randint(0, 3)
            image = image.resize((int(width * sizeRange[randSize]), int(height * sizeRange[randSize])))

            label = torch.tensor(float(label)).long()
            # 训练阶段图像增强，1、将关键区
        image = self.transforms(image)
        return label, image

    def removeLabel(self, image, bboxPath):
        '''移除目标'''
        width, height = image.size
        bboxes = self.readBboxes(image, bboxPath)
        for bbox in bboxes:
            bboxWidth = bbox[2] - bbox[0]
            bboxHeight = bbox[3] - bbox[1]
            if (bbox[2] + bboxWidth) <= width and (bbox[3] + bboxHeight) <= height:
                cropImage = image.crop((bbox[2], bbox[3], bbox[2] + bboxWidth, bbox[3] + bboxHeight)) 
                image.paste(cropImage,(bbox[0], bbox[1], bbox[2], bbox[3]))
                cropImage = None
            elif (bbox[0] - bboxWidth) >= 0 and (bbox[1] - bboxHeight) >= 0:
                cropImage = image.crop((bbox[0] - bboxWidth, bbox[1] - bboxHeight, bbox[0], bbox[1])) 
                image.paste(cropImage,(bbox[0], bbox[1], bbox[2], bbox[3]))
                cropImage = None
            else:
                image.paste((128,128,128),(bbox[0], bbox[1], bbox[2], bbox[3]))                
        bboxes = None
        return image

    def addLabel(self, image):
        '''增加目标,原来的图像没有标注为1'''
        width, height = image.size
        with open("/2020/data/usePhone/train/phoneFile.txt", "r") as phoneFile:
            images = phoneFile.readlines()
            random.shuffle(images)
            imagePath = images[10].replace("\n", "")

            bboxPath = imagePath.replace("JPEGImages", "labels").replace(".jpg", ".txt")
            labelImage = Image.open(imagePath)
            bboxes = self.readBboxes(labelImage, bboxPath)
            lam = np.random.beta(0.4, 0.4)
            for bbox in bboxes:
                bboxWidth = bbox[2] - bbox[0]
                bboxHeight = bbox[3] - bbox[1]
                bbox[0] = bbox[0] - bboxWidth * 2
                bbox[1] = bbox[1] - bboxHeight * 2
                bbox[2] = bbox[2] + bboxWidth * 2
                bbox[3] = bbox[3] + bboxHeight * 2
                # 如果截图大于原图三分之一则不进行添加目标
                if (bbox[2] - bbox[0]) > width  or (bbox[3] - bbox[1]) > height:
                    return image, False, 0
                # 如果截图小于原图的10分之一则不进行目标增加
                if (bbox[2] - bbox[0]) < width * 0.3 or (bbox[3] - bbox[1]) < height * 0.3:
                    return image, False, 0
                if bbox[2] > width + 10:
                    widthDiff = (bbox[2] - width)
                    bbox[2] = bbox[2] - widthDiff
                    bbox[0] = bbox[0] - widthDiff
                if bbox[3] > height + 10:
                    heightDiff = (bbox[3] - height)
                    bbox[3] = bbox[3] - heightDiff
                    bbox[1] = bbox[1] - heightDiff
                cropImage = labelImage.crop((bbox[0], bbox[1], bbox[2], bbox[3]))       # 0
                cropOriginImage = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))      # 1
                cropImage = Image.blend(cropOriginImage, cropImage, lam) # lam * cropImage + (1-lam) * cropOriginImage
                image.paste(cropImage,(bbox[0], bbox[1], bbox[2], bbox[3]))
                cropImage = None
            # image.save("/2020/data/usePhone/createImage/%s_%.3f.jpg" % (imagePath.split("/")[-1].split(".jpg")[0], lam))
            labelImage = None
            bboxes = None
            # if lam >= 0.5:
            #     lam = 0
            # else:
            #     lam = 1
            lam = 1 - lam 

            return image, True, lam

    def readBboxes(self, image, bboxPath):
        '''获取图像中的目标'''
        imageWidth, imageHeight = image.size
        bboxes = []
        i = 0
        with open(bboxPath, "r") as bboxesFile:
            lines = bboxesFile.readlines()
            for line in lines:
                line = line.replace("\n", "").split(" ")
                if len(line) < 4:
                    continue
                bbox_ = [float(line[1]) * imageWidth, float(line[2]) * imageHeight, 
                        float(line[3]) * imageWidth, float(line[4]) * imageHeight]
                bbox = [int(bbox_[0] - bbox_[2] * 0.6), int(bbox_[1] - bbox_[3] * 0.6), int(bbox_[0] + bbox_[2] * 0.6), int(bbox_[1] + bbox_[3] * 0.6)]
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[2] > imageWidth:
                    bbox[2] = imageWidth
                if bbox[3] > imageHeight:
                    bbox[3] = imageHeight
                bboxes.append(bbox)
        #     cropImage = image.crop((bbox[0], bbox[1], bbox[2], bbox[3])) 
        #     cropImage.save("/2020/data/usePhone/crop_image/%s_%d.jpg" % (bboxPath.split("/")[-1].split(".txt")[0], i))
        #     i = i + 1
        # print(imageWidth, imageHeight, bboxPath, bbox)
        return bboxes

    def __len__(self):
        return len(self.imagesLabels)

def phone_dataset_collate(batch):
    label_list, imgs_list = zip(*batch)
    pad_imgs_list = []
    h_list = [int(s.shape[1]) for s in imgs_list]
    w_list = [int(s.shape[2]) for s in imgs_list]
    max_h = np.array(h_list).max()
    max_w = np.array(w_list).max()
    labels = []
    for i in range(len(imgs_list)):
        img = imgs_list[i]
        img = torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)
        pad_imgs_list.append(img.numpy())
        labels.append(label_list[i].item())
   
    imgs = torch.from_numpy(np.array(pad_imgs_list))
    labels = torch.from_numpy(np.array(labels))
    return labels, imgs


def spliteUsePhoneData():
    phone_data = "/2020/data/usePhone/train/"
    usePhoneData = []
    noPhoneData = []
    usePhonePath = os.path.join(phone_data, "0_phone/JPEGImages")
    usePhoneFiles = os.listdir(usePhonePath)
    for usePhoneFile in usePhoneFiles:
        usePhoneData.append("0,0_phone/JPEGImages/" + usePhoneFile + "\n")
    
    noPhonePath = os.path.join(phone_data, "1_no_phone")
    noPhoneFiles = os.listdir(noPhonePath)
    for noPhoneFile in noPhoneFiles:
        noPhoneData.append("1,1_no_phone/" + noPhoneFile + "\n")
    totalData = []
    usePhoneNum = len(usePhoneData)
    noPhoneNum = len(noPhoneData)
    for i in range(usePhoneNum):
        totalData.append(usePhoneData[i])
        totalData.append(noPhoneData[i])
    for j in range(usePhoneNum, noPhoneNum):
        totalData.append(noPhoneData[j])
    
    random.shuffle(totalData)
    random.shuffle(totalData)
    random.shuffle(totalData)
    totalDataNum = usePhoneNum + noPhoneNum
    trainData = totalData[0: int(totalDataNum * 0.7)]
    valData = totalData[int(totalDataNum * 0.7):totalDataNum]
    with open("/2020/data/usePhone/train/trainFiles.txt", "w") as trainFile:
        trainFile.writelines(trainData)
    with open("/2020/data/usePhone/train/valFiles.txt", "w") as valFile:
        valFile.writelines(valData)
    print(usePhoneNum)
    print(noPhoneNum)
