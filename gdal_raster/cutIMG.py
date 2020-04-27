import cv2
import gdal
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
'''
使用gdal对图像进行切割，分成小块图
'''

class GRID:
    '''图像的操作类，可以实现对栅格数据的读取，写入以及裁剪等操作'''
    def __init__(self):
        pass
    #读图像文件
    def read_img(self,filename):
        dataset=gdal.Open(filename)                             # 打开文件
 
        im_width = dataset.RasterXSize                          # 栅格矩阵的列数
        im_height = dataset.RasterYSize                         # 栅格矩阵的行数
        bands = dataset.RasterCount                             # 获取波段数
 
        im_geotrans = dataset.GetGeoTransform()                 # 仿射矩阵，0，3表示左上角的坐标，1，5指示像元大小
        im_proj = dataset.GetProjection()                       # 地图投影信息
        # im_data = dataset.ReadAsArray(0,0,im_width,im_height)   # 将数据写成数组，对应栅格矩阵
 
        # del dataset                                             # 释放内存
        return im_width, im_height, im_proj, im_geotrans, dataset, bands
 
    #写文件，以写成tif为例
    def write_img(self,filename,im_proj,im_geotrans,im_data):
        #gdal数据类型包括
        #gdal.GDT_Byte, 
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64
 
        #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
 
        #判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape 
 
        #创建文件
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
 
        dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
        dataset.SetProjection(im_proj)                    #写入投影
 
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
 
        del dataset
    
    def cut_img(self, data_path, imgSize, out_path, save_name):
        '''图像的裁剪，对dataset进行部分读取转换为矩阵，然后进行保存.
            data_path为被剪切的数据；
            imSize为剪切的大小，这里为正方形;
            out_path为保存的位置;
            save_name为保存的名称。
        '''
        width, height, proj, geotrans, dataset, bands = self.read_img(data_path)  #读数据
        print(proj, geotrans)
        lon_x = geotrans[0]
        x_size = geotrans[1]
        lat_y = geotrans[3]
        y_size = geotrans[5]
        for i in range(width//imgSize):
            for j in range(height//imgSize):
                im_data = dataset.ReadAsArray(i*imgSize, j*imgSize, imgSize, imgSize)
                print(im_data.shape)
                ##------------这里需要转换下geotrans，因为切片之后仿射变换改变了。
                geotrans_ = (lon_x + i*imgSize*x_size, x_size, 0.0,
                    lat_y + j*imgSize*y_size, 0.0, y_size)
                self.write_img(out_path + '/{}_{}_{}.tif'.format(save_name, i, j),
                    proj, geotrans_, im_data)  ##写数据
        del im_data
        del dataset

    def write_img_png(self, type, out_path, im_data):
        '''将数据源保存为png图片'''
         #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
 
        #判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape 

        driver = gdal.GetDriverByName( 'MEM' )
        driver2 = gdal.GetDriverByName( 'PNG' )
        ds = driver.Create( '', im_height, im_width, 1, gdal.GDT_UInt16)
        if im_bands == 1:
            ds.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                ds.GetRasterBand(i+1).WriteArray(im_data[i])
        dataset = driver2.CreateCopy(out_path, ds, 0)
 
        del dataset

    def listdir_small_img(self, in_path, out_path):
        '''遍历文件夹，将.tif文件进行缩略图创建，名称为原名.jpg'''
        for filePath in glob.glob(in_path + "\\*.tif"):
            file_name = filePath.split("\\")[-1]
            out_name = file_name.split(".")[0]
            out_file = out_path + out_name + ".jpg"
            self.create_small_img(filePath, out_file)

    def create_small_img(self, in_path, out_path, img_size=600):
        '''生成缩略图,单个波段读取重采样数据，利用opencv进行三个波段合并，然后进行保存'''
        im_width, im_height, im_proj, im_geotrans, dataset, bands = self.read_img(in_path)
        print(im_width, im_height, im_geotrans, bands)
        ratio = im_width/im_height
        im_height_ = img_size if im_height>=im_width else img_size//ratio
        im_width_ = img_size if im_height<=im_width else img_size*ratio
        im_width_ = int(im_width_)
        im_height_ = int(im_height_)
        datas = []
        for i in range(bands):
            band = dataset.GetRasterBand(i+1)  
            data = band.ReadAsArray(0, 0, im_width, im_height, im_width_, im_height_)  
            datas.append(data)
        out_img = cv2.merge([datas[2], datas[1], datas[0]])
        cv2.imwrite(out_path, out_img)
        del dataset

    def tif2jpg(self, in_path, out_path):
        '''将tif转换为jpg
            1、首先获取数据基本信息，长、宽、波段等；
            2、将数据源转换为矩阵
            3、分波段读取数据，最后用opencv合并波段，写入jpg文件。
        '''
        im_width, im_height, im_proj, im_geotrans, dataset, bands = self.read_img(in_path)
        datas = []
        for i in range(bands):
            band = dataset.GetRasterBand(i+1)  
            data = band.ReadAsArray(0, 0, im_width, im_height)  
            datas.append(data)
        out_img = cv2.merge([datas[2], datas[1], datas[0]])
        cv2.imwrite(out_path, out_img)


if __name__ == "__main__":
    in_path = "D:\\Data\\GeoCache\\beyondMark\\P0005.tif"
    out_path = "D:\\Data\\GeoCache\\beyondMark\\raw_data\\P0005.jpg"
    grid = GRID()
    # grid.tif2jpg(in_path, out_path)
    grid.create_small_img(in_path, out_path, img_size=600)
