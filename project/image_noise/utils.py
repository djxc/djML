
import os
from osgeo import gdal, gdal_array
from pathlib import Path
from PIL import Image
from typing import Optional
import numpy as np


def get_tif_as_array(img_path, band_num=1):
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)    
    band = dataset.GetRasterBand(band_num).ReadAsArray()
    return band


def save_tif_result(out_tif, array, im_proj=None, im_geotrans=None, *, nodata: int = 65535) -> bool:
    """save the result tif"""
    if len(array.shape) == 3:
        (band_num, ref_height, ref_width) = array.shape
    else:
        band_num, (ref_height, ref_width) = 1, array.shape
    
    out_ds = gdal.GetDriverByName("GTiff").Create(
        out_tif,
        ref_width,
        ref_height,
        band_num,
        gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype),
        options=["COMPRESS=LZW"],
    )
    if im_geotrans is None:
        im_geotrans = (0.0, 1.0, 0.0, ref_height, 0.0, -1.0)
    out_ds.SetGeoTransform(im_geotrans)  # 写入仿射变换参数

    if im_proj is not None:
        out_ds.SetProjection(im_proj)  # 写入投影

    # write data
    if band_num == 1:
        out_ds.GetRasterBand(1).WriteArray(array)  # 写入数组数据
    else:
        for i in range(band_num):
            out_ds.GetRasterBand(i + 1).WriteArray(array[i, :, :])
            out_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    out_ds.FlushCache()
    return True

class BandSplit:
    def __init__(self, img_path: str, save_folder: str) -> None:
        self.img_path = img_path
        self.save_folder = save_folder
    
    def start(self):
        if Path(self.img_path).suffix.lower() in [".tif", ".tiff"]:
            self.__split_tif_band()
        else:
            self.__split_png_band()

    def __split_tif_band(self):
        """拆分波段"""
        img_name = Path(self.img_path).stem
        dataset = gdal.Open(self.img_path, gdal.GA_ReadOnly)
        bands = dataset.RasterCount  # 获取波段数
        for b in range(bands):
            band = dataset.GetRasterBand(b + 1).ReadAsArray()
            band_num = "{0:0=2d}".format(b + 1)
            band[band >= 32767] = 0
            save_path = os.path.join(self.save_folder, "{}_B{}.tif".format(img_name, band_num))
            save_tif_result(save_path, band, nodata=65535)
        del dataset

    def __split_png_band(self):
        """"""
        image = Image.open(self.img_path)
        imgs = image.split()
        image_name = Path(self.img_path).stem

        for i, img in enumerate(imgs):
            img.save(os.path.join(self.save_folder, "{}_b{}.png".format(image_name, i)))


def merge_tifs(
        tif_list: list,
        merge_name: str,
        *,
        pixel_size: Optional[float] = None,
        src_nodata: Optional[float] = None,
        dst_nodata: Optional[float] = None,
        separate_flag: bool = False,
        resample_alg: str = "near",
) -> bool:
    """using this to merge tif
    now only merge tif for small
    Args:
         tif_list: need merge tif
         merge_name: result tif
         pixel_size: output tif pixel size
         dst_nodata: input tif nodata
         src_nodata: output tif nodata
         separate_flag: stack or not
         resample_alg: change resample model

    Returns:
        bool

    Notes:
        https://gdal.org/python/osgeo.gdal-module.html#BuildVRTOptions

        how set reasmple algs
        https://gdal.org/programs/gdalwarp.html
    """
    # build vrt
    vrt_res_path = merge_name.replace(".tif", "_all.vrt")

    # get vrt parameter
    options_dict = {
        "separate": separate_flag,
        "resampleAlg": "near",  # gdalconst.GRA_NearestNeighbour
    }
    if pixel_size is not None:
        options_dict.update({"xRes": pixel_size, "yRes": pixel_size})
    if src_nodata is not None:
        options_dict.update({"srcNodata": src_nodata})
    if dst_nodata is not None:
        options_dict.update({"VRTNodata": dst_nodata})
    if resample_alg != "near":
        options_dict.update({"resampleAlg": resample_alg})
    if separate_flag:
        options_dict.update({"separate": separate_flag})

    # get python vrt options
    vrt_options = gdal.BuildVRTOptions(**options_dict)
    try:
        gdal.BuildVRT(vrt_res_path, tif_list, options=vrt_options)
    except Exception as e:
        print(e)
        return False

    # translate vrt to tif
    options = {
        "format": "GTiff",
        "creationOptions": ["BIGTIFF=YES", "COMPRESS=LZW"],
        "warpOptions": ["CUTLINE_ALL_TOUCHED=TRUE"],
    }
    try:
        gdal.Warp(merge_name, vrt_res_path, **options)
    except Exception as e:
        print(e)
        return False
    os.remove(vrt_res_path)
    return True

def merge_bands(band_list, save_path):
    dataset = gdal.Open(band_list[0])
    geoTrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()                       # 地图投影信息
    im_width = dataset.RasterXSize                          # 栅格矩阵的列数
    im_height = dataset.RasterYSize 
    result_array = np.zeros((len(band_list), im_height, im_width))
    for i, band_path in enumerate(band_list):
        if i > 0:
            dataset = gdal.Open(band_path)
        band_array = dataset.GetRasterBand(1).ReadAsArray()
        result_array[i] = band_array
    result_array = result_array.astype(dtype=np.uint16)
    save_tif_result(save_path, result_array, im_proj, geoTrans)

def change_tif_datatype(input_tif, output_tif, dataType="float32"):
    dataset = gdal.Open(input_tif)
    geoTrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()                       # 地图投影信息
    im_width = dataset.RasterXSize                          # 栅格矩阵的列数
    im_height = dataset.RasterYSize 
    bands = dataset.RasterCount
    result_array = np.zeros((bands, im_height, im_width))
    for band_index in range(bands):       
        band_array = dataset.GetRasterBand(band_index + 1).ReadAsArray()
        result_array[band_index] = band_array
    result_array = result_array.astype(dtype=dataType)
    save_tif_result(output_tif, result_array, im_proj, geoTrans)

if __name__ == "__main__":
    tif_list = [
            r"E:\Data\RS\result\GS18_20231014T204557_20231016T155715-1697287557-aa\patch\20231014T204557.tif"
        ]
    workspace = r"E:\Data\RS\remove_noise"
    for tif in tif_list:
        tif_name = Path(tif).stem
        save_folder = os.path.join(workspace, tif_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        band_split = BandSplit(tif, save_folder)
        band_split.start()


    # change_tif_datatype(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230310T191324_20230313T171720\GS18_MSS_L0S_20230310T191324_20230313T171720_rn_match.tif",
    #                     r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230310T191324_20230313T171720\GS18_MSS_L0S_20230310T191324_20230313T171720_rn_match_f.tif", dataType=np.float32)
    
    band_list = [
        r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230409T135419_20230410T173116_rs\GS18_MSS_L0S_20230409T135419_20230410T173116_rs_B01_pred18.tif",
        r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230409T135419_20230410T173116_rs\GS18_MSS_L0S_20230409T135419_20230410T173116_rs_B02_pred18.tif",
        r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230409T135419_20230410T173116_rs\GS18_MSS_L0S_20230409T135419_20230410T173116_rs_B03_pred18.tif",
        r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230409T135419_20230410T173116_rs\GS18_MSS_L0S_20230409T135419_20230410T173116_rs_B04_pred18.tif",
    ]

    # merge_bands(band_list, r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230409T135419_20230410T173116_rs\GS18_MSS_L0S_20230409T135419_20230410T173116_rs_pred.tif")

