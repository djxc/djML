from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .operateIMG import tensor_to_np
batch_size = 8

def MNISTData():
    # 1、ToTensor()把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray,
    # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    # 2、Normalize(),给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5],
            [0.5])
        ])
    train_dataset = datasets.MNIST(root="/2020/data/", train=True, transform=data_tf)
    test_dataset = datasets.MNIST(root="/2020/data/", train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def showData():
    train_loader, test_loader = MNISTData()
    index = 1
    for i, (images, labels) in enumerate(train_loader):
        img = tensor_to_np(images)
        plt.subplot(231 + i)
        plt.imshow(img)  # expects distorted color
        plt.title('label: ' + str(labels.item()))
        print(i, images.shape, labels.item(), img.shape)
        index += 1
        if index > 6:
            plt.savefig("/2020/numberDetect.jpg")
            break