# -*- coding: utf-8 -*-
import gdal


import os
import time
from pathlib import Path
from typing import List
from ast import literal_eval
from osgeo import gdal, ogr, osr

class CutTiffService:
    def __init__(self) -> None:
        self.output_path = r"d:/data/"

    def __parse_coords(self, coorsArray: List) -> List:
        """解析坐标,返回以下形式坐标
            target_coordinates = [
                (13541488.507040609, 4712868.672028078),
                (13541520.944112001, 4712868.672028078),
                (13541520.944112001, 4712910.253917306),
                (13541488.507040609, 4712910.253917306),
                (13541488.507040609, 4712868.672028078)
            ]
        """
        target_coordinates = []       
        coorsArray = coorsArray.split('|')
        coorsArray = list(map(literal_eval, coorsArray))
        target_coordinates.append(coorsArray)
        return target_coordinates   

    def __create_result_name(self, imgName: str):
        """生成结果文件名称"""
        current_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        newName = "cut_" + imgName + "_" + current_time_str + ".tiff"
        cutResultName = self.output_path + newName
        return cutResultName, newName
    
    def crop_tiff_by_vector_layer(self, input_path: str, coorsArray: str, imgName: str):
        """根据矢量图层进行裁剪栅格数据
            @param input_path 待裁剪的tif
            @param coorsArray 坐标字符串
            @param imgName tif名称
            @return 结果文件路径以及名称
        """
        cutResultName, newName = self.__create_result_name(imgName)     
        shp_path = cutResultName + "_cutpolygon.shp"
        try:
            coordinates = self.__parse_coords(coorsArray)
            self.__create_shp_vector_layer(coordinates, shp_path, input_path)
            options = gdal.WarpOptions(     
                creationOptions = ["BIGTIFF=YES", "COMPRESS=LZW"],
                cropToCutline = True, 
                cutlineDSName=shp_path,
                dstNodata=0
            )
            g = gdal.Warp(cutResultName, input_path, options = options)   
            g = None
            print('图像裁剪完成。')     
        except Exception as e:
            print("图像裁剪错误:{}".format(e))
        finally:
           self.__remove_shp(shp_path)

        return cutResultName, newName 
    
    def __remove_shp(self, shp_path):
        """移除shp文件"""
        if os.path.exists(shp_path):            
            shp_file_path = Path(shp_path)
            shp_root = shp_file_path.parent
            shp_name = shp_file_path.stem
            shp_type_list = [".shp", ".dbf", ".prj", ".shx"]
            for shp_type in shp_type_list:
                shp_type_name = shp_name + shp_type
                shp_type_path = os.path.join(shp_root, shp_type_name)
                if os.path.exists(shp_type_path):
                    os.remove(shp_type_path)

    def __create_shp_vector_layer(self, coordinates: List, shp_file_path: str, origin_raster_path: str):
        """创建shape文件，坐标系和待切割tif一致
            @param coordinates 坐标数组
            @param shp_file_path shp文件保存路径
            @parma origin_raster_path 待切割tif路径
        """

        # 创建多边形图层
        raster_ds = gdal.Open(origin_raster_path)
        sr = osr.SpatialReference()
        sr.ImportFromWkt(raster_ds.GetProjection())

        driver = ogr.GetDriverByName('ESRI Shapefile')
        polygon_ds = driver.CreateDataSource(shp_file_path)
        layer = polygon_ds.CreateLayer('polygon', geom_type=ogr.wkbPolygon, srs=sr)

        # 创建多边形几何对象
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in coordinates[0]:
            ring.AddPoint(coord[0], coord[1])
        ring.CloseRings()
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        # 创建要素并设置几何对象
        feature_defn = layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        feature.SetGeometry(polygon)

        # 将要素添加到图层
        layer.CreateFeature(feature)
        feature = None
        polygon_ds = None

# 读取要切的原图
in_ds = gdal.Open("chengdu-xinglong-lake.tif")
print("open tif file succeed")

# 读取原图中的每个波段
in_band1 = in_ds.GetRasterBand(1)
in_band2 = in_ds.GetRasterBand(2)
in_band3 = in_ds.GetRasterBand(3)

# 定义切图的起始点坐标(相比原点的横坐标和纵坐标偏移量)
offset_x = 100  # 这里是随便选取的，可根据自己的实际需要设置
offset_y = 100

# 定义切图的大小（矩形框）
block_xsize = 400  # 行
block_ysize = 400  # 列

# 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
out_band2 = in_band2.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
out_band3 = in_band3.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)

# 获取Tif的驱动，为创建切出来的图文件做准备
gtif_driver = gdal.GetDriverByName("GTiff")

# 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
out_ds = gtif_driver.Create('clip.tif', block_xsize, block_ysize, 3, in_band1.DataType)
print("create new tif file succeed")

# 获取原图的原点坐标信息
ori_transform = in_ds.GetGeoTransform()
if ori_transform:
    print (ori_transform)
    print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
    print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))

# 读取原图仿射变换参数值
top_left_x = ori_transform[0]  # 左上角x坐标
w_e_pixel_resolution = ori_transform[1] # 东西方向像素分辨率
top_left_y = ori_transform[3] # 左上角y坐标
n_s_pixel_resolution = ori_transform[5] # 南北方向像素分辨率

# 根据反射变换参数计算新图的原点坐标
top_left_x = top_left_x + offset_x * w_e_pixel_resolution
top_left_y = top_left_y + offset_y * n_s_pixel_resolution

# 将计算后的值组装为一个元组，以方便设置
dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])

# 设置裁剪出来图的原点坐标
out_ds.SetGeoTransform(dst_transform)

# 设置SRS属性（投影信息）
out_ds.SetProjection(in_ds.GetProjection())

# 写入目标文件
out_ds.GetRasterBand(1).WriteArray(out_band1)
out_ds.GetRasterBand(2).WriteArray(out_band2)
out_ds.GetRasterBand(3).WriteArray(out_band3)

# 将缓存写入磁盘
out_ds.FlushCache()
print("FlushCache succeed")

# 计算统计值
# for i in range(1, 3):
#     out_ds.GetRasterBand(i).ComputeStatistics(False)
# print("ComputeStatistics succeed")

del out_ds

print("End!")

