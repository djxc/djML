import arcpy
import math
"""
author: djxc
date: 2019-09-18
xtent函数，输入值为矢量点数据，输出值为栅格泰森多边形
1、首先将矢量点数据转换为栅格，栅格值为矢量点的C字段的值
2、然后利用栅格计算器，将值为nodata的设为255。小于255的栅格即为原始点数据
3、将栅格数据转换为矩阵
"""
# 将栅格转换为矩阵，并把小于255的栅格(原始点数据)保存在points中，point保存栅格点的X、Y坐标以及值
def Raster2Matrix(path):
    raster = arcpy.Raster(path)
    matrix = arcpy.RasterToNumPyArray(path)
    w, h = matrix.shape
    points = []
    # 获取点的位置，保存在points    
    for i in range(w):
        for j in range(h):
            if matrix[i][j] < 255:
                point = [i, j, matrix[i][j]]
                points.append(point)
    return matrix, points, raster

# 遍历栅格像素，通过公式计算每个像素与对应点的值，值最大的即为当前像元的类别。并设置象元类别值为点的C值
def XTENT(matrix, points):
    w, h = matrix.shape
    for i1 in range(w):
        for j1 in range(h):
            m = 0
            v = 0
            for x in range(len(points)):
                va = math.pow(points[x][2] * 10, a) - k * math.sqrt(math.pow((i1-points[x][0]), 2) + math.pow((j1-points[x][1]), 2))
                if va > m:
                    m = va
                    v = x
            matrix[i1][j1] = points[v][2]
    return matrix

# 将矩阵转换为栅格，并设置栅格的空间参考、像素大小等；保存栅格数；最后删除raster清理内存
def Matrix2Raster(raster, matrix, out_path, outRasterName):
    img_SR = arcpy.Describe(raster).spatialReference 
    lowerLeft = arcpy.Point(raster.extent.XMin, raster.extent.YMin) #左下角点坐标
    cellWidth = raster.meanCellWidth #栅格宽度
    cellHeight = raster.meanCellHeight

    raster_out = arcpy.NumPyArrayToRaster(matrix, lowerLeft, cellWidth, cellHeight)
    arcpy.DefineProjection_management(raster_out, img_SR)
    raster_out.save(out_path + "\\" + outRasterName)

    del raster

if __name__ == "__main__":
#    C = 1000
    a = 1
    k = 1

    inRaster = arcpy.GetParameterAsText(0)
    outRaster = arcpy.GetParameterAsText(1)
    outRasterName = arcpy.GetParameterAsText(2)

    matrix, points, raster = Raster2Matrix(inRaster)
    Matrix2Raster(raster, XTENT(matrix, points), outRaster, outRasterName)









