import os
import random


def split_data(workspace):
    """将数据拆分为训练集和验证集
        1、首先将数据打乱，拆分为10份
        2、选取其中2份为验证集，其他的为训练集；然后再选择其他两份为验证集，其他为训练集，依次循环5次
    """
    img_folder = os.path.join(workspace, "img")
    label_folder = os.path.join(workspace, "label")
    img_list = os.listdir(img_folder)
    random.shuffle(img_list)

    ten_size = int(len(img_list) / 10)
    for i in range(5):
        verify_img_list = img_list[i * ten_size * 2 : (i + 1) * ten_size * 2]
        train_img_list = img_list[:i * ten_size * 2]
        train_img_list2 = img_list[(i + 1) * ten_size * 2:]
        train_img_list.extend(train_img_list2)
        with open(os.path.join(workspace, "train_{}.csv".format(i + 1)), "w") as img_path_file:
            for train_img in train_img_list:
                img_path = os.path.join(img_folder, train_img)
                label_path = os.path.join(label_folder, train_img)
                img_path_file.write("{},{}\n".format(img_path, label_path))

        with open(os.path.join(workspace, "verify_{}.csv".format(i + 1)), "w") as img_path_file:
            for verify_img in verify_img_list:
                img_path = os.path.join(img_folder, verify_img)
                label_path = os.path.join(label_folder, verify_img)
                img_path_file.write("{},{}\n".format(img_path, label_path))

