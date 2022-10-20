# 利用pytesseract进行文字识别
# 1、首先利用Image读取图像，并将其装换为灰度图
# 2、二值化，设置阈值#

import os
import numpy as np
import pytesseract
from PIL import Image

def readLocationFromImage(imagePath, fname, outPath):
    image =Image.open(imagePath)
    image1 = image.convert('L')
    w, h = image.size[0], image.size[1]
    table = []
    for i in range(256):
        if i < 200:
            table.append(0)
        else:
            table.append(1)

    photo = image1.point(table, "1")
    # photo.save("/2020/location/ocr_" + imagePath.split("/")[-1])

    # 每个10行判断图像当前5行内的均值是否大于200
    # print(w, h)
    lineEndY = []
    for i in range(0, h, 5):
        box = (0, i, w, i + 5)
        image_crop = image1.crop(box)
        image_crop = np.array(image_crop)
        if np.mean(image_crop) > 230:
            lineEndY.append(i + 4)
            # print(np.mean(image_crop), i)
    
    # print(lineEndY)
    i = 0
    lastEndY = 0
    xy_cooridinates = []
    for ey in lineEndY:   
        if ey - lastEndY > 15:
            box = (0, lastEndY, w, ey + 5)
            image_crop = photo.crop(box)
            image_crop_array = np.array(image_crop)
            lastEndY = ey
            if np.mean(image_crop_array) <= 220:
                code = pytesseract.image_to_string(image_crop)
                codes = code.split("\n")
                i = i + 1   
                x, y = parseXY(codes, i)    
                if x > 0:
                    xy_cooridinates.append([i, x, y])
                elif x == 0:
                    continue
                else:
                    print('can not detect xy coordinate from %s', fname)
                    return
    if len(xy_cooridinates) == 0:
        return
    # 将数据保存下来
    with open(os.path.join(outPath, f.split(".")[0] + ".csv"), "w") as file_object:
        file_object.write("id, x, y\n")
        for xy in xy_cooridinates:
            file_object.write(str(xy[0]) + ", " + str(xy[1]) + ", " + str(xy[2]) + "\n")
                # image_crop.save('/2020/location/test_' + fname.replace(".JPG", "") + str(i) + '.png')  # 写入图片


def parseXY(xyArray, i):
    ''' 解析坐标值，如果解析成功则将其返回，否则返回-1
        @return （x, y）
    '''
    x = -1
    y = -1
    for c in xyArray:
        c = c.replace(" ", "")
        if c is not "\x0c" and len(c) > 0:
            c = c.replace(",", ".")
            c = c.replace(":", ".")
            c = c.replace(";", ".")
            cs = c.split(".")
            if len(cs[0]) > len(str(i)):
                x = cs[0][1:-1] + "." + cs[1]
                y = cs[2] + "." + cs[3]                
            else:            
                x = cs[1] + "." + cs[2]
                y = cs[3] + "." + cs[4]
    if float(x) == -1:
        x = float(0)
        y = float(0)
    else:
        x = float(x)
        y = float(y)
    return (x, y)
    

if __name__ == "__main__":
    image_root = "/2020/location/originImage"
    files = os.listdir(image_root)
    # readLocationFromImage("/2020/location/1ocr_dfmz.JPG")       
    for f in files:
        fPath = os.path.join(image_root, f)
        print(fPath)
        readLocationFromImage(fPath, f, "/2020/location/")       
