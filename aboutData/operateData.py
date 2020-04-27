# -*- coding: utf-8 -*-
"""
本文件是对图像以及文件的操作，如：文件重命名、文件移动、图像值修改以及图像旋转等操作
Created on Mon Sep 30 17:14:02 2019

@author: dj_jx
"""

import cv2
import os, shutil
from PIL import Image

def renameFile(path):
    fileList = os.listdir(path)
    for file in fileList:
        name = file.split('.')[0]
        name_ = int(name)
        newName = ''
        if name_ < 10:
            newName = '0000' + str(name_)
        else:
            newName = '000' + str(name_)
        os.rename(os.path.join(path,file),os.path.join(path, newName+".jpg"))
    #    print(name)

def moveImgRename(path, toPath):
    '''遍历文件夹下的文件夹，将img.png、label.png复制重命名到新的文件夹下'''
    fileList = os.listdir(path)
    index = 0
    for file in fileList:
        if os.path.isdir(path + file):
            imgFile = path + file + "/img.png"
            shutil.copyfile(imgFile, toPath + "img/%03d.png"%index)
            labelFile = path + file + "/label.png"
            shutil.copyfile(labelFile, toPath + "label/%03d_mask.png"%index)
            print(toPath + "img/%03d.png"%index)
            print(toPath + "label/%03d_mask.png"%index)
            print("*********************")
            index = index + 1                  

def changeLabel(labelPath, fileName):
    '''将label图片通道分离，只取一个通道，将大于0的值设为255'''
    img_x = cv2.imread(labelPath)
    img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
    x, y = img_x.shape
    for i in range(x):
        for j in range(y):
            if img_x[i, j] > 0:
                img_x[i, j] = 255
    im = Image.fromarray(img_x)
    im.save("%03d.png"%fileName)

def changeIMGValue(path):
    '''将label的值改为255'''
    index = 0
    fileList = os.listdir(path)
    for file in fileList:
        if os.path.isdir(path + file):            
            labelFile = path + file + "/label.png"
            changeLabel(labelFile, index)
            index = index + 1

def tif2png(path):
    fileList = os.listdir(path)
    fileName = 14
    for file in fileList:
        print(file)
        img_x = cv2.imread(file)
        im = Image.fromarray(img_x)
        im.save("%03d.png" % fileName)
        fileName = fileName + 1

def roateIMG(path):
    '''将图像进行旋转，增加数据集，生成更多可能性的样本'''
    image = cv2.imread(path) 
    
    (h, w) = image.shape[:2] #10
    center = (w // 2, h // 2)  #11
    M = cv2.getRotationMatrix2D(center, -90, 1.0) #15
    rotated = cv2.warpAffine(image, M, (w, h)) #16
    
    M1 = cv2.getRotationMatrix2D(center, 180, 1.0) #15
    rotated1 = cv2.warpAffine(image, M, (w, h)) #16  


if __name__ == "__main__":
    path = './data/jsonCarStreet/'
    # moveImgRename(path, './data/streetCarMark/')
    # changeIMGValue(path)
    tif2png("./test")