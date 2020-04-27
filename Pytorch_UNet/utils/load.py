#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    '''列出文件夹内所有的文件，去掉文件名后缀名四位，获取文件名并以list形式返回'''
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """遍历ids， 为每个id生成tuples：(id, 0)，(id, 1)"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """根据传递的scale对图像进行裁剪，返回图像"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """获取所有的图像以及对应的mask以(img, mask)形式返回"""

    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)     # .jpg

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.png', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.png')
    mask = Image.open(dir_mask + id + '_mask.png')
    return np.array(im), np.array(mask)
