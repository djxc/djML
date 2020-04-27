import os, shutil
from labelme2coco import labelme2coco
from osgeo import gdal

rootPath = '../../Data/StreetCar/'
files = os.listdir(rootPath)
num = 0
jsonFile = []
driver=gdal.GetDriverByName('PNG')
for f in files:
    name, suffix = f.split('.')
    if suffix == 'json':
        path = rootPath + name + '.tif'
        # shutil.copyfile(imgFile, toPath + "img/%03d.png"%index)
        # jsonFile.append(path)
        ds = gdal.Open(path)
        dst_ds = driver.CreateCopy('./' + name + '.png', ds)
        print(path)
        num += 1
dst_ds = None
ds = None
print(num)
# labelme2coco(jsonFile)