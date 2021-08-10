# 数据读取与处理


import sys
import torch
import torchvision
from PIL import Image
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import os
import time
import numpy as np
import hashlib
import zipfile, tarfile, requests
from scipy.io import loadmat


CURRENT_IMAGE_PATH = "/2020/"

# D:\Data\机器学习\fashion-MNIST /2020/data/
def load_data_fashion_mnist(batch_size, resize=None, root="/2020/data/"):
    '''采用torchvision进行图像数据的读取
    '''
    trans = []
    # 数据的转换，resize等操作
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def create_data(num_examples, num_inputs):
    '''生成随机数据
        1、首先利用numpy生成随机数据，范围为0-1，然后将其转换为tensor格式    
        2、然后利用生成的数据与true_w相乘，在加上true_b得到label  
        3、为了模拟数据的随机性，为每个标签添加了0.01的误差
        @param num_examples 为生成数据的个数  
        @param num_inputs 为数据的维度  
        @return features 数据, labels 标签
    '''
    true_w = [2, -3.4]
    true_b = 4.2
    # 特征以及权重，b数据类型要一致
    # 随机生成1000个2维，范围在0-1之间的样本数据，转换为torch格式向量
    features = torch.from_numpy(np.random.normal(
        0, 1, (num_examples, num_inputs))).to(torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 标签数据随机增加0.01的误差
    labels += torch.from_numpy(np.random.normal(0,
                                                0.01, size=labels.size())).to(torch.float32)
    return features, labels



def sgd(params, lr, batch_size):  # 优化函数
    '''优化函数'''
    for param in params:
        param.data -= lr * param.grad / batch_size


def data_iter(batch_size, features, labels):
    '''分批读取数据，每批包含batch_size个数据'''
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def showData(x, y, save=False):
    '''显示matplotlib绘制的图片，由于docker中不方便直接显示，这里将其保存在指定文件下显示'''
    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend(loc='upper left')
    plt.scatter(x, y, c='',
                        alpha=1.0, linewidth=1.0, marker='o', edgecolors='yellow',
                        s=55, label='test set')
    if save:
        plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg", dpi=600) 
    else:
        plt.show()


DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['banana-detection'] = (DATA_URL + 'banana-detection.zip', '5de26c8fce5ccdea9f91267273464dc968d20d72')
# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
DATA_HUB['voc2012'] = (DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')
DATA_HUB['voc2007'] = (DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
def download(name, cache_dir=os.path.join('/2020', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签。"""
    data_dir = download_extract('banana-detection')
    csv_fname = os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir,
                             'bananas_train' if is_train else 'bananas_val',
                             'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def read_voc_images(voc_dir, is_train=True):
    """读取VOC影像数据以及label"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels, fnames = [], [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        label = Image.open(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png')).convert('RGB')
        label = np.array(label, dtype="float32")
        label = torch.from_numpy(label).permute(2, 0, 1)    
        labels.append(label)
        fnames.append(fname)
    return features, labels

def voc_colormap2label():
    """颜色表到类索引值的映射
        1、首先创建一个256^3长度的数组，用来保存映射关系
        2、遍历颜色表，将颜色表的三颜色按照一定规则相加，变为唯一的数据，值为类的索引值
        @return colormap2label 三原色到类的映射
    """
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """三个通道的图像像素值映射为类的索引
        1、首先将图像数据进行通道转换为长宽通道顺序
        2、然后根据映射关系，返回映射好的数据
        @param colormap 原始图像
        @param colormap2label 映射关系
        @return 像素值为类索引的数据
    Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """裁剪图像以及label"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class FlattenLayer(nn.Module):
    '''继承nn.Module为一个模型或一个网络层
        该层网络
    '''

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集。"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (
            f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels, fnames = read_voc_images(voc_dir, is_train=is_train)       
        self.features = [
            self.normalize_image(feature)
            for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        '''对图像进行标准化'''
        img = img.float()
        img = (img- img.mean()) / img.std()
        # print(img.max(), img.min(), img.mean(), img.std())
        return img
        # return self.transform(img.float())

    def filter(self, imgs):
        return [
            img for img in imgs if (img.shape[1] >= self.crop_size[0] and
                                    img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
def load_data_voc(batch_size, crop_size, pred=False):
    """Load the VOC semantic segmentation dataset."""
    # voc_dir = download_extract('voc2012',
    #                                os.path.join('VOCdevkit', 'VOC2012'))
    voc_dir = "/2020/data/VOCdevkit/VOC2012"
    num_workers = 2
    train_iter = None
    if pred == False:
        print("load train data")
        train_iter = torch.utils.data.DataLoader(
            VOCSegDataset(True, crop_size, voc_dir), batch_size, shuffle=True,
            drop_last=True, num_workers=num_workers)
    print("load test data")
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter

class ITCVDDataset(torch.utils.data.Dataset):
    '''
    '''
    def __init__(self, is_train, root_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.root_dir = root_dir
        self.images = self.list_files()
        # self.features = self.normalize_image(feature)
        print('read ' + str(len(self.images)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def list_files(self):
        ''''''
        images = os.listdir(os.path.join(self.root_dir, "clipIMG"))
        return images


    def __getitem__(self, idx):
        '''读取图像以及label
            1、读取图像数据， 图像尺寸为1024*1024,原始图像为5616,3744
            2、图像的名中获取label数据名以及该图像在原始图像中的位置
            3、label中的数据需要减去宽度以及高度，去除小于0的值以及大于该图像尺寸的值
        '''
        img_name = self.images[idx]
        
        feature = torchvision.io.read_image(os.path.join(self.root_dir, "clipIMG", f'{img_name}'))
        # feature = Image.open(os.path.join(self.root_dir, "clipIMG", f'{img_name}')).convert('RGB')
        # feature = np.array(feature)
        # feature = torch.from_numpy(feature).permute(2, 0, 1)
        label_name = img_name.split("_")[0]
        label = load_ITCVD_label(os.path.join(self.root_dir, "GT", f'{label_name}'), img_name, feature)
       
        # return (feature, label)

        return (feature.float(), label)

    def __len__(self):
        return len(self.images)

def load_ITCVD_label(label_path, img_name, feature):
    '''加载label'''
    label_name = img_name.split("_")[0]
    img_off_width = int(img_name.split("_")[1])
    img_off_height = int(img_name.split("_")[2].split('.')[0])
    label = loadmat(label_path)
    label = label["x" + label_name]
    label = label[:, 0:4]
    # print(label)

    # label_ = label - np.array([img_off_height, img_off_width, img_off_height, img_off_width])
    label = label - np.array([img_off_width, img_off_height, img_off_width, img_off_height])


    # print(label)
    imageWH = np.array([feature.shape[2], feature.shape[1], feature.shape[2], feature.shape[1]])
    label = label / imageWH
    label = label[np.all(label >= 0, axis=1)]
    label = label[np.all(label <= 1, axis=1)]
    test = np.zeros((label.shape[0], 1))
    label = np.c_[test,label]
    # label = torch.from_numpy(label.astype(float))
    # print(img_name, label.shape)
    return label



def load_data_bananas(batch_size):
    """加载香蕉检测数据集。"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

def load_data_ITCVD(batch_size):
    ''' 加载ITCVD数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = torch.utils.data.DataLoader(
        ITCVDDataset(True, "/2020/data/ITCVD/ITC_VD_Training_Testing_set/Training"), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)
        # , collate_fn=ITCVD_dataset_collate)
    # print("load test data")
    test_iter = []

    test_iter = torch.utils.data.DataLoader(
        ITCVDDataset(False, "/2020/data/ITCVD/ITC_VD_Training_Testing_set/Testing"), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter


def ITCVD_dataset_collate(batch):
    images = []
    bboxes = None
    for img, box in batch:
        # print( box)
        images.append(img.numpy())
        # bboxes.append(box.tolist())
        if bboxes is None:
            bboxes = np.array([box])
        else:
            bboxes = np.append(bboxes, np.array([box]), axis=0)
        # bboxes.append(box)
    # print(bboxes)
    bboxes = torch.from_numpy(bboxes)
    images = torch.from_numpy(np.array(images))
    print(bboxes.shape, images.shape)
    return images, bboxes
