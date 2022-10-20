'''
计算每个栅格到最近道路的高度差
1、道路进行栅格化，道路栅格大小可以设为10m
2、对dem数据进行重采样，设为10m大小与道路一致
3、窗口运算，每个像素向外扩充，寻找道路像素，然后计算像素点与道路的差
'''
#
# @author djxc
# @date 2019-12-03#

from osgeo import gdal, osr
import numpy as np

def OpenData(rasterPath, streetPath):
    """根据输入的栅格路径位置，打开数据"""
    data = gdal.Open(rasterPath)

    rasterW = data.RasterXSize  # 栅格的宽度，即为列数
    rasterH = data.RasterYSize  # 栅格的高度，即为行数
    
    street_data = gdal.Open(streetPath)
    rasterW1 = street_data.RasterXSize  # 栅格的宽度，即为列数
    rasterH1 = street_data.RasterYSize      # 栅格的高度，即为行数
    
    raster_data = data.ReadAsArray(0, 0, rasterW, rasterH)  # 栅格dataset转换为矩阵
    street_data1 = street_data.ReadAsArray(0, 0, rasterW1, rasterH1)
    shape_ = raster_data.shape
    shape1 = street_data1.shape
    x = (shape1[0] if (shape_[0] > shape1[0]) else shape1[0])
    y = (shape1[1] if (shape_[1]>shape1[1]) else shape1[1])   
    return raster_data, street_data1, x, y, data

def calculate(raster_data, street_data, x, y):
    """循环遍历raster
        1、循环每个原始栅格
        2、从1开始，以1为间隔扩展
        3、循环每个扩展的像素，判断是否为1，为1跳出循环
    """   
    # newRaster = raster_data

    newRaster =np.zeros((x,y))
    print(newRaster.shape)
    for i in range(x):
        for j in range(y):
            if raster_data[i][j] < 2000:
                diff_h = extendBy1([i, j], raster_data, street_data, x, y)
                newRaster[i][j] = diff_h
                # print(diff_h, raster_data[i][j])
        print(i)
    return newRaster

def extendBy1(point, raster, street, maxX_, maxY_):
    """从1开始，以1为间隔扩展
    point为当前点的xy坐标;maxX_为最大高度；maxY_为最大宽度
    """
    temp = []
    maxX = maxX_
    maxY = maxY_   
    pointX = point[0]
    pointY = point[1]
    for step in range(1, 100):
        for stepX in range(pointX - step, pointX + step):
            if stepX > 0 and stepX < maxX:
                if pointY - step >= 0:       
                    if street[stepX][pointY - step] > 0:
                        temp.append([stepX, pointY - step])
                if (pointY + step) < maxY:
                    if street[stepX][pointY + step] > 0:
                        temp.append([stepX, pointY + step])                     

        for stepY in range(pointY - step + 1, pointY + step - 1):
            if stepY > 0 and stepY < maxY:
                if (pointX - step) >= 0:
                    if street[pointX - step][stepY] > 0:
                        temp.append([pointX - step, stepY])
                if (pointX + step) < maxX:
                    if street[pointX + step][stepY] > 0:
                        temp.append([pointX + step, stepY])

        origin = raster[pointX][pointY]
        if len(temp):
            min_ = abs(origin - raster[temp[0][0]][temp[0][1]])
            for i in range(len(temp)):
                tem_min = abs(origin - raster[temp[i][0]][temp[i][1]])
                # if tem_min > 700:
                #     print('dj',origin, raster[temp[i][0]][temp[i][1]])
                if min_ > tem_min:
                    min_ = tem_min
            return min_
    return 20000

def exportTiff(cols, rows, array, inputDataset):
    im_geotrans = inputDataset.GetGeoTransform()  #仿射矩阵
    im_proj = inputDataset.GetProjection() #地图投影信息
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create("/home/djxc/2019/test2.tif", cols, rows, 1, gdal.GDT_Int16)        # 要注意数组的数据类型
    # 括号中两个0表示起始像元的行列号从(0,0)开始
    outRaster.SetGeoTransform(im_geotrans)
    # 获取数据集第一个波段，是从1开始，不是从0开始
    outRaster.SetProjection(im_proj)   
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)

    outband.FlushCache()  

if __name__ == "__main__":
    rasterPath = "/home/djxc/2019/Data/laoshan_dem2.tif"
    streetPath = "/home/djxc/2019/Data/street_cut.tif"
    raster_data, street_data ,x, y, rasterData = OpenData(rasterPath, streetPath)
    print(x, y)
    outArray = calculate(raster_data, street_data, x, y)
    exportTiff(y + 3, x + 1, outArray, rasterData)  
