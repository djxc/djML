'''
图像的卷积操作
@date 2020-10-19
@author small dj
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms import transforms

from DJ_u_net.unet import Unet


def convolution(img, kernel):
    '''单波段的卷积运算'''
    h, w = img.shape
    img_new = np.zeros((h - 3, w - 3), dtype=np.float)
    for i in range(h - 3):
        for j in range(w - 3):
            img_new[i, j] = np.sum(np.multiply(img[i:i + 3, j:j + 3], kernel))
    img_new = img_new.clip(0, 255)
    return np.array(img_new).astype('uint8')


def convolve(img, fil, mode='same'):
    '''多波段的卷积运算，首先分离波段，然后每个波段进行卷积，
    然后将结果进行合并'''
    if mode == 'fill':
        h = fil.shape[0] // 2
        w = fil.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w), (0, 0)), 'constant')
    conv_b = convolution(img[:, :, 0], fil)  # 然后去进行卷积操作
    conv_g = convolution(img[:, :, 1], fil)
    conv_r = convolution(img[:, :, 2], fil)
    dstack = np.dstack([conv_b, conv_g, conv_r])  # 将卷积后的三个通道合并
    return dstack  # 返回卷积后的结果


def create_kernel(kernelName=""):
    if kernelName == "":
        # 卷积核
        fil = np.array([[-1, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 1]])
    elif kernelName == "mean":
        # 平滑均值滤波
        fil = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1 / 9, 1 / 9, 1 / 9]])
    elif kernelName == "guss_mean":
        # 高斯平滑均值滤波
        fil = np.array([[1/16, 2/16, 1/16],
                        [2/16, 4/16, 2/16],
                        [1 / 16, 2 / 16, 1 / 16]])
    elif kernelName == "sharp":
        # 锐化滤波
        fil = np.array([[-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]])
    elif kernelName == "soble1":
        # 竖状边缘检测
        fil = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    elif kernelName == "soble2":
        # 横状边缘检测
        fil = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    return fil


def tensor_to_np(tensor):
    '''tensor格式转换为numpy的图像格式，将第一个纬度去掉，然后跳转波段与行列顺序'''
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def img2tensor(img):
    '''将opencv格式矩阵转换为tensor，然后扩展纬度，torch中需要[batch, channel, width, height]
    调整图像大小
    '''
    img = cv2.resize(img, (256, 256))
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((256, 256))
    ])
    img2_new = x_transforms(img)
    img2_new = img2_new.unsqueeze(dim=0)
    return img2_new


def conv_torch(img):
    '''通过pytorch的内部方法进行卷积运算、上下采样以及激活函数操作。
    pytorch的卷积核为随机的卷积核，会根据反向传播算法进行更新参数。'''
    torchConv = nn.Conv2d(3, 3, (3, 3))
    batchNorm = nn.BatchNorm2d(3)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool2d(2)

    conv_img = torchConv(img)
    normal_img = batchNorm(conv_img)
    relu_img = relu(normal_img)
    maxpool_img = maxpool(relu_img)
    return maxpool_img


def hidden_layer(in_ch, out_ch, img):
    ''''''
    hiddenLayer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
    return hiddenLayer(img)


def showResult():
    img = cv2.imread("/home/djxc/test3.jpg")
    b, g, r = cv2.split(img)
    # 原始图像， matplotlib显示图像是bgr需要修改为rgb
    img2 = cv2.merge([r, g, b])
    plt.subplot(231)
    plt.imshow(img2)  # expects distorted color
    plt.title('origin')

    # torch计算的卷积
    torchConv = nn.Conv2d(3, 3, (3, 3))
    img2_new = torchConv(img2tensor(img2))
    plt.subplot(232)
    plt.imshow(tensor_to_np(img2_new))
    plt.title('torch conv')

    # 标准化将图像转换为0-1
    batchNorm = nn.BatchNorm2d(3)
    batchNorm_img = batchNorm(img2_new)
    fil2 = create_kernel("soble2")
    new_img2 = convolve(img2, fil2)

    plt.subplot(233)
    plt.imshow(tensor_to_np(batchNorm_img))
    plt.title("normal")

    # 激活函数
    relu = nn.ReLU(inplace=True)
    relu_img = relu(batchNorm_img)
    plt.subplot(234)
    plt.imshow(tensor_to_np(relu_img))
    plt.title("relu")

    # 池化，改变图像尺寸，保留主要信息，减少运算量
    maxpool = nn.MaxPool2d(2)
    pool_img = maxpool(relu_img)
    plt.subplot(235)
    plt.imshow(tensor_to_np(pool_img))
    plt.title("pool")

    # pool_img1 = conv_torch(img2tensor(img2))
    # hidden_layer_img1 = hidden_layer(3, 3, img2tensor(img2))
    # hidden_layer_img1 = maxpool(hidden_layer_img1)
    # hidden_layer_img2 = hidden_layer(3, 3, hidden_layer_img1)
    # hidden_layer_img2 = maxpool(hidden_layer_img2)
    # hidden_layer_img3 = hidden_layer(3, 3, hidden_layer_img2)
    # # hidden_layer_img3 = maxpool(hidden_layer_img3)

    # hidden_layer_img4 = hidden_layer(3, 3, hidden_layer_img3)
    # # hidden_layer_img4 = maxpool(hidden_layer_img4)
    # hidden_layer_img = hidden_layer(3, 3, hidden_layer_img4)
    model = Unet(3, 3)
    hidden_layer_img = model(img2tensor(img2))
    plt.subplot(236)
    plt.imshow(tensor_to_np(hidden_layer_img))
    plt.title("2 conv")
    plt.show()
