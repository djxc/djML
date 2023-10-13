
import os
import numpy as np
from pathlib import Path
from osgeo import gdal, gdal_array
import scipy.signal as ss
import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.fftpack as fft


def save_matrix_as_img(img_data, img_path):
    """"""
    im = Image.fromarray(img_data)
    im.save(img_path)

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


def med_filter(img_path, save_floder):
    """最小值滤波"""
    img_name = Path(img_path).stem
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1).ReadAsArray()
    heigh, width = band.shape
    print(heigh, width, img_name)
    new_band = ss.medfilt2d(band, [5,5])
    save_path = os.path.join(save_floder, "{}_minfilter5.tif".format(img_name))
    save_tif_result(save_path, new_band, nodata=65535)

def min_filter(img_path, save_floder):
    """最小值滤波"""
    now_time = time.time()
    img_name = Path(img_path).stem
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1).ReadAsArray()
    heigh, width = band.shape
    print(heigh, width, img_name)
    start_i = 3000
    start_j = 2000
    padding_size = 1
    new_data = np.zeros((heigh, width))
    step_size = 1 # padding_size * 2 + 1
    for i in range(start_i + padding_size, heigh - padding_size - 1, step_size):
        for j in range(start_j + padding_size, width - padding_size - 1, step_size):
            data = band[i - padding_size:i + padding_size + 1, j - padding_size: j + padding_size + 1]
            min_data = np.min(data)
            max_data = np.max(data)         
            new_data[i, j] = min_data
            if j > 5000:
                break
        if i > 5000:
            break    
    save_path = os.path.join(save_floder, "{}_min_filter5_04.tif".format(img_name))
    save_tif_result(save_path, new_data, nodata=65535)
    print("spend time: {}s".format(time.time() - now_time))

def get_window_png(img_path, save_path):
    now_time = time.time()
    img_name = Path(img_path).stem
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1).ReadAsArray()
    data = band[500:2000, 3500:5000]
    save_matrix_as_img(data, save_path)

def cv2_remove_noise(img_path, save_path):
    """"""
    # img = cv2.imread(img_path, 0)
    # print(img.shape)
    # dst = cv2.fastNlMeansDenoising(img, None, 5,7,21)
    # cv2.imwrite(save_path, dst)
    moon_data = plt.imread(img_path) #ndarray
    #plt.figure(figsize=(12,11))    #调整图片显示大小
    #plt.imshow(moon_data,cmap = 'gray')  #autumn
    print(moon_data.shape)  #二维  黑白
    moon_fft = fft.fft2(moon_data)
    print(moon_fft.min())
    moon_fft2 = np.where(np.abs(moon_fft)>1e3, 0, moon_fft)
    moon_ifft = fft.ifft2(moon_fft2)  #逆变化
    moon_result = np.real(moon_ifft)
    plt.figure(figsize=(12,11))
    plt.imshow(moon_result,cmap = 'gray')    
    plt.imsave(save_path, moon_result)
    plt.show()

def get_part_tif(img_path, save_path, start_line, end_line):
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1).ReadAsArray()
    data = band[start_line:end_line, :]
    save_tif_result(save_path, data)

def fill_invalid_value(img_path):
    """将无效值进行邻域均值填充， 300"""
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1).ReadAsArray()
    count = np.sum(band > 300)
    padding_size = 2
    heigh, width = band.shape
    # for i in range(heigh):
    #     for j in range(width):
    #         data = band[i, j]         
    #         if data > 300:
    #             data_array = band[i - padding_size:i + padding_size + 1, j - padding_size: j + padding_size + 1]
    #             band[i, i] = np.min(data_array)
    #             print(data, band[i, i])
    print(count)
    # save_tif_result(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20221221T024552_20230104T122441_B04_part_valid.tif", band)

def subtract_base_value(img_path, band_num=1, base_value=105):
    """减去夜间海里的值，默认夜间海里值为100"""    
    band = get_tif_as_array(img_path, band_num)
    band = band - base_value
    return band

def get_tif_as_array(img_path, band_num=1):
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)    
    band = dataset.GetRasterBand(band_num).ReadAsArray()
    return band

def remove_vertical(band, save_path):
    """纵向上滤波弱化条纹
        1、每二十行为一组，每列按照奇偶分为两类，将异常值赋值为每类中间十行的均值
    """
    heigh, width = band.shape
    new_band = np.copy(band)
    print(heigh, width)
    cell_num = 28
    half_cell_num = int(cell_num / 2)
    threshold_num = 0.2
    for i in range(0, heigh - half_cell_num, half_cell_num):
        data = band[i:i + cell_num, :]
        data1 = data[::2, :]        # 偶数行
        data2 = data[1::2, :]       # 奇数行
        # 按照列取均值
        data1_mean = np.median(data1, axis=0)
        data2_mean = np.median(data2, axis=0)
        # 设置阈值为均值的1/3，如果超过1/3则将其设置为其值与均值的均值
        data1_threshold = data1_mean * threshold_num
        data2_threshold = data2_mean * threshold_num
        data1 = np.where(data1 > (data1_mean + data1_threshold), (data1 * 0.2 + data1_mean * 0.8), data1)
        data1 = np.where(data1 < (data1_mean - data1_threshold), (data1 * 0.2 + data1_mean * 0.8), data1)
        data2 = np.where(data2 > (data2_mean + data2_threshold), (data2 * 0.2 + data2_mean * 0.8), data2)
        data2 = np.where(data2 < (data2_mean - data2_threshold), (data2 * 0.2 + data2_mean * 0.8), data2)
        new_band[i:i+cell_num:2, :] = data1
        new_band[i+1:i+cell_num:2, :] = data2    
    save_tif_result(save_path, new_band)


def find_vertical(img_path):
    band = get_tif_as_array(img_path)
    #设定画布。dpi越大图越清晰，绘图时间越久
    fig = plt.figure(figsize=(4, 4), dpi=400)
    #导入数据
    len_size = 500
    col_num1 = 2000
    col_num2 = 2001
    x = list(np.arange(1, len_size + 1))
    y_v = band[:len_size, col_num1]
    y_v1 = band[:len_size, col_num2]
    #绘图命令
    plt.plot(x, y_v, lw=0.5, ls='-', c='b', alpha=0.1)
    plt.plot(x, y_v1, lw=0.5, ls='-', c='r', alpha=0.1)
    #show出图形
    plt.show()


def find_horizontal(img_path):
    band = get_tif_as_array(img_path)
    #设定画布。dpi越大图越清晰，绘图时间越久
    fig = plt.figure(figsize=(4, 4), dpi=400)
    #导入数据
    len_size = 500
    col_num = 200
    row_num1 = 104
    row_num2 = 105
    x = list(np.arange(1, len_size + 1))
    y_h = band[row_num1, col_num:col_num + len_size]
    y_h1 = band[row_num2, col_num:col_num + len_size]
    #绘图命令
    plt.plot(x, y_h, lw=0.5, ls='-', c='b', alpha=0.1)
    plt.plot(x, y_h1, lw=0.5, ls='-', c='r', alpha=0.1)
    #show出图形
    plt.show()

        
def remove_noise(img_path, noise_path, save_path):
    noise_heigh = 12288
    band = get_tif_as_array(img_path)
    heigh, width = band.shape
    cut_heigh = noise_heigh
    if heigh < noise_heigh:
        cut_heigh = heigh
    noise_band = get_tif_as_array(noise_path)
    new_data = band[:cut_heigh, :].astype(dtype=np.int16)
    new_data[:cut_heigh, :] = band[:cut_heigh, :] - noise_band[:cut_heigh, :]
    save_tif_result(save_path, new_data)

def remove_he(img_path):
    """横条纹的去除
        1、需要监测到横条纹，在相同的行像素差趋势相同则认为是横条纹，进行均值滤波，否则可能是地物特征不进行处理。
        2、取20行， 50列为一组，求出改组的列的均值，设置阈值，如果每行超过该均值个数大于90%则认为改行为噪音，需要重新赋值
    """
    band = get_tif_as_array(img_path)
    heigh, width = band.shape
    new_band = np.copy(band)
    print(heigh, width)
    row_num = 28
    col_num = 100
    threshold_num = 0.05
    for row in range(0, heigh - row_num, int(row_num/2)):
        for col in range(0, width - col_num, col_num):
            data = band[row : row + row_num, col : col + col_num]            
            # 按照列取均值
            data_mean = np.median(data, axis=0)
            # 设置阈值为均值的1/3，如果超过1/3则将其设置为其值与均值的均值
            data_threshold = data_mean * threshold_num
            recal = 0
            for i in range(row_num):
                data_sub = data[i]
                if np.sum(data_sub > (data_mean + data_threshold)) > 80 or np.sum(data_sub < (data_mean - data_threshold)) > 80:
                    new_band[row + i, col : col + col_num] = data[i] * 0.2 + data_mean * 0.8   
                    recal = recal + 1
        if row % 100 == 0:
            print("finish row:{}; recal:{}".format(row, recal))     
    tif_path = Path(img_path)
    save_path = os.path.join(tif_path.parent, "{}_rmh.tif".format(tif_path.stem))
    save_tif_result(save_path, new_band)
    

def draw_horizontal(img_path):
    """绘制直方图"""
    img_arr=gdal_array.LoadFile(img_path)
    #计算频率
    min_num = np.min(img_arr)
    max_num = 500 # np.max(img_arr)
    gray=range(min_num, max_num)
    hist=[]
    #迭代图像中的不同波段
    #迭代器迭代出每个像素
    im_iter=img_arr.flat
    #对像素的灰度值进行排序
    rank=np.sort(im_iter)
    #找出灰度值的顺序号，不包括最大的，searchsorted（a，v）在数组a中插入数组v
    #返回插入位置下标组成的数组
    index=np.searchsorted(rank, gray)
    #附加最大值到最后
    n=np.append(index, [len(im_iter)])
    #错位相减得到每个灰度值的像素个数
    item_hist=n[1:]-n[:-1]
    #作为列表元素附加到hist中，这个元素即代表某个波段的频率数据
    hist.append(item_hist)
    print(len(hist[0]), min_num, len(gray) + 1)
    fig = plt.figure(figsize=(4, 4), dpi=400)    
    x = list(np.arange(min_num, max_num))   
    #绘图命令
    plt.plot(x, hist[0], lw=0.5, ls='-', c='r', alpha=0.1)
    #show出图形
    plt.show()   

def change_geotransform(img_path, save_path):
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    bands = dataset.RasterCount  # 获取波段数
    if bands == 0:
        return
    band1 = dataset.GetRasterBand(1).ReadAsArray()
    save_tif_result(save_path, band1, nodata=65535)
    del dataset


def remove_noise_by_fft():
    """利用傅里叶变换去除条纹"""
    tif_path = r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230222T101418_20230227T155433\GS18_MSS_L0S_20230222T101418_20230227T155433_B01_rm.tif"
    img = get_tif_as_array(tif_path)
    tif_path_p = Path(tif_path)
    save_path = os.path.join(tif_path_p.parent, "{}_fft_rmv5.tif".format(tif_path_p.stem))

    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)

    # fshift[:5070, 3950:3970] = 0
    # fshift[5166:, 3950:3970] = 0
    # # fshift[:5070, 3950] = 0
    # # fshift[5166:, 3970] = 0
    # # fshift[5120:, :] = 0

    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = np.fft.ifft2(f_ishift)
    # img_back = np.abs(img_back)
    # minv, maxv = np.amin(img, (0, 1)), np.amax(img, (0, 1))
    # img_back = cv2.normalize(img_back, None, alpha=minv, beta=maxv, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    # save_tif_result(save_path, img_back, nodata=65535)

    # # convert image to float and do dft saving as complex output
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)

    # create a mask, center square is 1, remaining all 0
    rows, cols = img.shape
    mask = np.zeros((rows,cols,2),np.uint8)
    dft_shift[:5070, 3950:3970] = 1
    dft_shift[5166:, 3950:3970] = 1
    fshift = dft_shift * mask
    back_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(back_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # re-normalize to 16-bits
    minv, maxv = np.amin(img, (0, 1)), np.amax(img, (0, 1))
    img_back = cv2.normalize(img_back, None, alpha=minv, beta=maxv, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    # save tif
    save_tif_result(save_path, img_back, nodata=65535)
    sm_save_path = os.path.join(tif_path_p.parent, "{}_fft_sm5.tif".format(tif_path_p.stem))
    save_tif_result(sm_save_path, fshift, nodata=65535)


def fft_test():
    
    img = get_tif_as_array(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230222T101418_20230227T155433\GS18_MSS_L0S_20230222T101418_20230227T155433_B01.tif")

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))

    rows,cols = img.shape
    crow,ccol = rows//2,cols//2
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131)
    plt.imshow(img,cmap = 'gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    plt.imshow(img_back,cmap = 'gray')
    plt.title('Image After HPF')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    plt.imshow(img_back)
    plt.title('Result in JET')
    plt.xticks([])
    plt.yticks([])

    plt.show()

def test_dft():
    radius = 500
    tif_path = r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230310T191324_20230313T171720\GS18_MSS_L0S_20230310T191324_20230313T171720_B01_trans_rm.tif"
    img = get_tif_as_array(tif_path)
    tif_path_p = Path(tif_path)
    save_path = os.path.join(tif_path_p.parent, "{}_fft_{}.tif".format(tif_path_p.stem, radius))

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    rows,cols = img.shape
    crow,ccol = rows//2,cols//2
    print(crow, ccol)
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-radius:crow+radius,ccol-radius:ccol+radius] = 1
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    save_tif_result(save_path, img_back, nodata=65535)

    # plt.subplot(121),plt.imshow(img,cmap = 'gray')
    # plt.title('Input Image'),plt.xticks([]),plt.yticks([])
    # plt.subplot(122),plt.imshow(img_back,cmap = 'gray')
    # plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
    # plt.show()


def get_vertial_noise(band_path):
    tif_path = Path(band_path)
    save_path = os.path.join(tif_path.parent, "{}_rmv.tif".format(tif_path.stem))
    valid_band = subtract_base_value(band_path, band_num=1)
    rmv_tif_path = remove_vertical(valid_band, save_path)

if __name__ == "__main__":
    """
        1、拆分波段
        2、填充无效值
        3、利用夜间拍摄数据获取噪声数据：减去均值，默认夜间海水均值为100
        4、噪声在垂直方向上有规律，一强一弱，利用均值弱化横条纹，得到近似于竖条纹的噪音
        5、用图像减去竖条纹噪音，实现竖条纹的去除
    """
    # min_filter(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230306T020432_20230307T104849_B04.tif", r"E:\Data\RS\remove_noise")
    # get_window_png(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230306T020432_20230307T104849_B04_rmnV.tif", r"E:\Data\RS\remove_noise\demo3.png")
    # cv2_remove_noise(r"E:\Data\RS\remove_noise\demo3.png", r"E:\Data\RS\remove_noise\demo3_rn1.png")
    workspace = r"E:\Data\RS\remove_noise"
    tif_list = [
        # "GS18_MSS_L0S_20230306T020432_20230307T104849",
        # "GS18_MSS_L0S_20230310T191324_20230313T171720",

        # "GS18_MSS_L0S_20230606T151928_20230613T141928",
        # "GS18_MSS_L0S_20230723T152556_20230809T114326",
        # "GS18_MSS_L0S_20230728T162416_20230808T184209",
        # "GS18_MSS_L0S_20230815T141255_20230818T165547",
        # "GS18_MSS_L0S_20230827T101806_20230831T105943",
    ]
    for tif in tif_list:
        for band in range(4):
            tif_path = os.path.join(workspace, tif, "{}_B0{}.tif".format(tif, band + 1))
            save_tif_path = os.path.join(workspace, tif, "{}_B0{}_part.tif".format(tif, band + 1))

            get_part_tif(tif_path, save_tif_path, 5000, 9200)
    # get_part_tif(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230310T191324_20230313T171720\GS18_MSS_L0S_20230310T191324_20230313T171720_B01_trans.tif", 
    #              r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230310T191324_20230313T171720\GS18_MSS_L0S_20230310T191324_20230313T171720_B01_trans_part.tif", 0, 2000)
    # for i in range(4):
    #     tif_path = r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230306T020432_20230307T104849\GS18_MSS_L0S_20230306T020432_20230307T104849_B0{}.tif".format(i + 1)
    #     tif_path_v = Path(tif_path)
    #     save_path = os.path.join(tif_path_v.parent, "{}_trans.tif".format(tif_path_v.stem))
    #     change_geotransform(tif_path, save_path)

    # fill_invalid_value(r"E:/Data/RS/remove_noise/GS18_MSS_L0S_20230103T011822_20230109T091202_B04_valid_rmv.tif")
    # find_vertical(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20221221T024552_20230104T122441_B04_part_ver1.tif")
    # find_horizontal(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20221221T024552_20230104T122441_B04_part_ver1.tif")
    # draw_horizontal(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20221221T024552_20230104T122441_B04_part_ver1.tif")
    # remove_he(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230306T020432_20230307T104849_B04_rmn_part.tif")

    # 用夜间拍摄的海洋作为去噪数据
    # 1、将样例数据清洗，将异常值进行临近均值填充
    tif_path = r"E:\Data\RS\L0S\GS18_MSS_L0S_20230227T192300_20230306T155532\GS18_MSS_L0S_20230227T192300_20230306T155532.tif"
    tif_path_val = Path(tif_path)

    # remove_noise(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230222T101418_20230227T155433\GS18_MSS_L0S_20230222T101418_20230227T155433_B01.tif",
    #              r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230103T011822_20230109T091202\GS18_MSS_L0S_20230103T011822_20230109T091202_B01_trans_noise_resampleV.tif",
    #              r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230222T101418_20230227T155433\GS18_MSS_L0S_20230222T101418_20230227T155433_B01_rm.tif")

    # fft_test()
    # test_dft()
    # remove_noise_by_fft()

    get_vertial_noise(r"E:\Data\RS\remove_noise\GS18_MSS_L0S_20230908T185311_20230915T131421\GS18_MSS_L0S_20230908T185311_20230915T131421_B01.tif")


