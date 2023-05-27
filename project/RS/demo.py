from osgeo import gdal
import numpy as np

def write_img_png(out_path, im_data):
    '''将数据源保存为png图片'''
        #判断栅格数据的数据类型
    # if 'int8' in im_data.dtype.name:
    #     datatype = gdal.GDT_Byte
    # elif 'int16' in im_data.dtype.name:
    #     datatype = gdal.GDT_UInt16
    # else:
    #     datatype = gdal.GDT_Float32
    datatype = gdal.GDT_Byte

    #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 

    driver = gdal.GetDriverByName('MEM')
    driver2 = gdal.GetDriverByName('PNG')
    ds = driver.Create('', im_height, im_width, im_bands, datatype)
    if im_bands == 1:
        ds.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
    else:
        for i in range(im_bands):
            ds.GetRasterBand(i+1).WriteArray(im_data[i])
    dataset = driver2.CreateCopy(out_path, ds, 0)
    del dataset

def get_data(img_path):
    """"""        
    ds = gdal.Open(img_path, gdal.GA_ReadOnly)

    alphaband = ds.GetRasterBand(1).GetMaskBand() # 获取的掩膜，有值的为0，其他的nodata

    tx = 90
    ty = 75
    tilesize = 256
    tilebands = 4

    rx = 1903
    ry = 5166
    rxsize = 343
    rysize = 343
    wxsize = 256
    wysize = 256
    dataBandsCount = 3
    wx = 0
    wy = 0

    tilefilename = r"E:\\其他\\tiles\\75.png"
    # result = np.zeros((tilebands, rxsize, rysize))
    # for i in range(1, dataBandsCount + 1):
    #     band = ds.GetRasterBand(i)
    #     band_data = band.ReadAsArray(rx, ry, rxsize, rysize)
    #     result[i-1] = band_data
    #     print(band_data)

    # result[-1] = alphaband.ReadAsArray(rx, ry, rxsize, rysize)
    # write_img_png(tilefilename, result)

    mem_drv = gdal.GetDriverByName('MEM')
    out_drv = gdal.GetDriverByName('PNG')
    dstile = mem_drv.Create('', tilesize, tilesize, tilebands - 1, gdal.GDT_Byte)

    data = ds.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize,
                                band_list=list(range(1, dataBandsCount + 1)))
    print(data.shape)
    dstile.WriteRaster(wx, wy, wxsize, wysize, data,
                               band_list=list(range(1, dataBandsCount + 1)))  
    alpha = alphaband.ReadRaster(rx, ry, rxsize, rysize, wxsize, wysize)
    dstile.WriteRaster(wx, wy, wxsize, wysize,
                               alpha, band_list=[tilebands])
    out_drv.CreateCopy(tilefilename, dstile, strict=0)


if __name__ == "__main__":
    tilefilename = r"E:\\其他\\tiles\\75.png"
    get_data(r"E:\china_cloudless.vrt")
