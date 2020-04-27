#
# @author djxc#

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from operateDataSet import  FaceLandmarksDataset
from myTransform import Rescale, ToTensor, RandomCrop

# Ignore warnings
import warnings

def getLandMarks(n):
    '''获取脸姿态点数据,需要输入一个数据的位置'''
    landmarks_frame = pd.read_csv('../data/faces/face_landmarks.csv')
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    return img_name, landmarks

def show_landmarks(image, landmarks):
    """显示图像以及特征点，需要输入图像以及对应的特征点坐标"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.01)  # pause a bit so that plots are updated

def testFacelandmarks(face_dataset):
    '''利用自己创建的类获取数据'''
    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break

def transformIMG(face_dataset):
    scale = Rescale(256)    
    crop = RandomCrop(128)  
    composed = transforms.Compose([Rescale(256),
                                RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = face_dataset[61]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)

    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # plt.ion()   # interactive mode
    # plt.figure()
    # img_name, landmarks = getLandMarks(20)
    # show_landmarks(io.imread(os.path.join('../data/faces/', img_name)),
    #            landmarks)
    # plt.show()    
    face_dataset = FaceLandmarksDataset(csv_file='../data/faces/face_landmarks.csv',
                                    root_dir='../data/faces/')
    # testFacelandmarks(face_dataset)
    transformIMG(face_dataset)