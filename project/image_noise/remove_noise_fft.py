# 傅里叶变换处理图像噪声#
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from image_noise.utils import get_tif_as_array, save_tif_result

class RemoveNosiseFFT:

    def remove_vertical_nosize(self, img_path):
        """去除竖条纹
            1、计算频谱图，竖条纹是水平方向中线较亮的线
            2、将水平方向较亮的线抹除
            3、反向傅里叶变换
        """        
        img = get_tif_as_array(img_path)
        dshift, minv, maxv = self.__cal_ms_cv(img)
        heigh, width = img.shape
        half_heigh = heigh // 2
        half_width = width // 2

        mask = np.ones((heigh, width, 2),np.uint8)
        mask[half_heigh - 1:half_heigh + 1, :half_width - 10] = 0
        mask[half_heigh - 1:half_heigh + 1, half_width + 10:] = 0
        fshift = dshift * mask
        img_back = self.__reverse_fft(fshift, minv, maxv)            
        print(img_back.dtype, np.max(img_back), np.min(img_back), np.mean(img_back))
        self.show_imgs([img, img_back], ["input image", "remove vertical noise"])

    def remove_horizontal_nosize(self, img_path):
        """去除横条纹
            1、计算频谱图，横条纹是垂直方向中线较亮的线
            2、将垂直方向较亮的线抹除
            3、反向傅里叶变换
        """        
        img = get_tif_as_array(img_path)
        img[img<0] = 0
        dshift, minv, maxv = self.__cal_ms_cv(img)
        heigh, width = img.shape
        half_heigh = heigh // 2
        half_width = width // 2

        width_length = 5
        center_offset = 100
        mask = np.ones((heigh, width, 2),np.uint8)
        mask[:half_heigh - center_offset, half_width - width_length:half_width + width_length] = 0
        mask[half_heigh + center_offset:, half_width - width_length:half_width + width_length] = 0
        fshift = dshift * mask
        print(fshift[:, :, 0].shape)
        # self.show_imgs([img, fshift[:, :, 0]], ["input image", "remove vertical noise"])
        img_back = self.__reverse_fft(fshift, minv, maxv)      
        tif_path_p = Path(img_path)
        save_path = os.path.join(tif_path_p.parent, "{}_rmv7.tif".format(tif_path_p.stem))
        print(np.max(img_back), np.min(img_back))
        save_tif_result(save_path, img_back)

    def remove_vh_noise(self, img_path):
        """移除竖条纹以及横条纹"""
        img = get_tif_as_array(img_path)
        dshift, minv, maxv = self.__cal_ms_cv(img)
        heigh, width = img.shape
        half_heigh = heigh // 2
        half_width = width // 2
        width_length = 20
        heigh_length = 1
        center_offset = 10
        mask = np.ones((heigh, width, 2),np.uint8)
        mask[half_heigh - heigh_length:half_heigh + heigh_length, :half_width - center_offset] = 0
        mask[half_heigh - heigh_length:half_heigh + heigh_length, half_width + center_offset:] = 0
        mask[:half_heigh - center_offset, half_width - width_length:half_width + width_length] = 0
        mask[half_heigh + center_offset:, half_width - width_length:half_width + width_length] = 0
        fshift = dshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        img_back = cv2.normalize(img_back, None, alpha=minv, beta=maxv, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        print(img_back.dtype, np.max(img_back), np.min(img_back), np.mean(img_back))     
        tif_path_p = Path(img_path)
        save_path = os.path.join(tif_path_p.parent, "{}_rmvh03.tif".format(tif_path_p.stem))
        save_tif_result(save_path, img_back)


    def save_img_ms(self, img_path):
        """保存频谱图"""
        magnitude_spectrum = self.__cal_ms(img_path)
        tif_path_p = Path(img_path)
        save_path = os.path.join(tif_path_p.parent, "{}_ms.tif".format(tif_path_p.stem))
        save_tif_result(save_path, magnitude_spectrum)

    def draw_img_ms(self, img_path):
        """绘制频率图"""
        magnitude_spectrum, minv, maxv = self.__cal_ms_np(img_path)        
        # plt.subplot(121),plt.imshow(img,cmap = 'gray')
        # plt.title('Input Image'),plt.xticks([]),plt.yticks([])
        # plt.subplot(122),
        plt.imshow(magnitude_spectrum,cmap = 'gray')
        plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
        plt.show()
    
    def show_imgs(self, img_data_list, title_list):
        """绘制两个图像进行对比"""
        plt.subplot(121),plt.imshow(img_data_list[0], cmap = 'gray')
        plt.title(title_list[0]),plt.xticks([]),plt.yticks([])
        plt.subplot(122),
        plt.imshow(img_data_list[1],cmap = 'gray')
        plt.title(title_list[1]),plt.xticks([]),plt.yticks([])
        plt.show()
    
    def __reverse_fft(self, fshift: np.ndarray, minv: int, maxv: int) -> np.ndarray:
        """反向傅里叶变换"""
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        print(np.max(img_back), np.min(img_back), minv, maxv)

        img_back = cv2.normalize(img_back, None, alpha=minv, beta=maxv, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)   
        return img_back


    def __cal_ms_np(self, tif_path: str) -> np.ndarray:
        """计算频谱图
            1、通过numpy计算二维离散的傅里叶变换
            2、将零频率移动到频谱中心
        """
        img = get_tif_as_array(tif_path)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        magnitude_spectrum = magnitude_spectrum.astype(np.uint16)
        minv, maxv = np.amin(img, (0, 1)), np.amax(img, (0, 1))
        return magnitude_spectrum, minv, maxv
    
    def __cal_ms_cv(self, img: np.ndarray) -> np.ndarray:
        """计算频谱图
            1、通过numpy计算二维离散的傅里叶变换
            2、将零频率移动到频谱中心
        """
        dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)
        minv, maxv = np.amin(img, (0, 1)), np.amax(img, (0, 1))
        return fshift, minv, maxv
       
