import cv2
import os
import numpy as np

size = 16
ySize = 8
outPath = "/2020/panoramaSplitIMG/"

def splitParamIMG(imagePath):
    '''对全景影像进行切片'''
    img = cv2.imread(imagePath)
    height, width, bands = img.shape
    print(height, width, bands)
    imageName = imagePath.split("/")[-1].replace(".jpg", "")
    print(imageName)
    imgSavePath = os.path.join(outPath, imageName)
    if not os.path.exists(imgSavePath):
        os.makedirs(imgSavePath)
    for i in range(size):
        for j in range(ySize):
            img_part = img[int(height/ySize) * j:int(height/ySize) * (j+1), int(width / size) * i:int(width/size) * (i+1)]
            cv2.imwrite(outPath + imageName + "/" + imageName + str(i+1) + str(j+1) + ".jpg", img_part)          
            print(outPath + imageName + "/" + imageName + str(i+1) + str(j+1) + ".jpg")

def cutIMG():
    '''剪切图片'''
    img = cv2.imread("/2020/2021_08_05_zjhsk_02.jpg")
    height, width, bands = img.shape
    img_part = img[:, 0:2680]
    img_part1 = img[:, 2680:]
    print(height, width, bands)
    cv2.imwrite(outPath + "part1.jpg", img_part)  
    cv2.imwrite(outPath + "part2.jpg", img_part1) 

def appendIMG():
    '''两幅图像拼接'''
    img = cv2.imread("/2020/panoramaSplitIMG/part2.jpg") 
    img1 = cv2.imread("/2020/panoramaSplitIMG/part1.jpg")
    img1 = np.fliplr(img1)
    image3 = np.hstack([img, img1])
    cv2.imwrite("/2020/panoramaSplitIMG/all1.jpg", image3) 

if __name__ == "__main__":
    # images = ["/2020/2021_08_05_zjhsk_02.jpg"]
    image_folder = "/2020/全景影像"
    images = os.listdir(image_folder)
    for image in images:
        splitParamIMG(os.path.join(image_folder, image))
        print(image)
    # cutIMG()
    # appendIMG()