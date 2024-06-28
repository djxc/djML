# 显示图像

import os

workspace = r"D:\Data\MLData\rs_road\算法赛道1高分辨率遥感数据道路提取初赛数据集\数据集\train"
img_list = os.listdir(os.path.join(workspace, "img"))
label_list = os.listdir(os.path.join(workspace, "label"))