# 显示图像

import os
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt

from config import WORKSPACE

img_folder = os.path.join(WORKSPACE, "img")
label_folder = os.path.join(WORKSPACE, "label")
img_list = os.listdir(img_folder)
img_list.sort()
label_list = os.listdir(label_folder)
label_list.sort()

def show_img_and_label(img_path: str, label_path: str):
    plt.figure(figsize=(10, 20))
    img = Image.open(img_path)
    plt.subplot(1, 2, 1)
    plt.title("img")
    plt.imshow(img)

    label = Image.open(label_path)
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label)

def save_label_rgb(label_path: str):
    """将label保存为rgb格式"""
    label = Image.open(label_path)
    label_np = np.array(label)
    label_np_tmp = np.where(label_np > 0, 255, 0)
    image_array = np.zeros((512, 512, 3), dtype=np.uint8)
    image_array[:, :, 0] = label_np_tmp
    image_array[:, :, 1] = label_np_tmp
    image_array[:, :, 2] = label_np_tmp

    label_img = Image.fromarray(image_array)
    rgb_label_name = Path(label_path).stem + "_rgb.png"
    label_img.save(os.path.join(WORKSPACE, "label_show", rgb_label_name))


for i, img in enumerate(img_list):
    img_path = os.path.join(img_folder, img)
    label_path = os.path.join(label_folder, label_list[i])
    # show_img_and_label(os.path.join(img_folder, img), os.path.join(label_folder, label_list[i]))
    save_label_rgb(label_path)

