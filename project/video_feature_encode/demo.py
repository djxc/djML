import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from PIL import Image

def load_npy():
    """用numpy读取文件，每个文件为250*2408*1*1矩阵"""
    depthmap = np.load(r'E:\Data\MLData\视觉特征编码\train\train_feature\000000.npy')    #使用numpy载入npy文件
    depthmap = np.squeeze(depthmap, -1)
    depthmap = np.squeeze(depthmap, -1)
    print(depthmap.shape)
    plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
    plt.colorbar()                   #添加colorbar
    # plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
    plt.show()   

def statistic_label(label_path):
    label_statis_result = {}
    with open(label_path) as label_file:
        labels = json.loads(label_file.read())
        # print(labels)
        for label in labels:
            cls = labels[label]
            if cls not in label_statis_result:
                label_statis_result[cls] = 1
            else:
                label_statis_result[cls] = label_statis_result[cls] + 1
    print(label_statis_result)

def save_npy_as_image(label_path):
    """"""
    data_root = os.path.join(Path(label_path).parent, "train_feature")
    with open(label_path) as label_file:
        labels = json.loads(label_file.read())
        # print(labels)
        for i, label in enumerate(labels):
            cls = labels[label]
            data_file = os.path.join(data_root, label)
            npy_to_image(data_file, cls)      

def npy_to_image(npy_path, cls):
    """用numpy读取文件，每个文件为250*2408*1*1矩阵"""
    npy_name = Path(npy_path).stem
    save_root = r"E:\Data\MLData\视觉特征编码\train\train_image"
    depthmap = np.load(npy_path)    #使用numpy载入npy文件
    depthmap = np.squeeze(depthmap, -1)
    depthmap = np.squeeze(depthmap, -1)
    print(npy_path, cls, np.max(depthmap), np.min(depthmap))
    cls_path = os.path.join(save_root, cls)
    cls_path1 = Path(cls_path)
    cls_path1.mkdir(parents=True, exist_ok=True)
    
    w = depthmap.shape[0]
    h = depthmap.shape[1]
    dpi = 300
    fig = plt.figure(figsize=(w/50,h/50),dpi=dpi)
    axes=fig.add_axes([0,0,1,1])
    axes.set_axis_off()
    axes.imshow(depthmap)
    plt.savefig(os.path.join(cls_path, npy_name + ".jpg"), bbox_inches='tight', dpi=dpi)
    plt.clf()
    plt.close(fig)

def split_train_verfy(label_path, radio=0.3):
    """"""
    root_path = Path(label_path).parent
    data_root = os.path.join(root_path, "train_feature")
    label_list = []
    with open(label_path) as label_file:
        labels = json.loads(label_file.read())
        for i, label in enumerate(labels):
            cls = labels[label]
            data_file = os.path.join(data_root, label)
            label_list.append("{}, {}".format(data_file, cls))
    np.random.shuffle(label_list)
    label_size = len(label_list)
    verify_list = ""
    train_list = ""
    for i, image in enumerate(label_list):
        if i <= label_size * radio:
            verify_list = verify_list + image + ",\n"
        else:
            train_list = train_list + image + ",\n"

    with open(os.path.join(root_path, "train.csv"), "w+", encoding="utf-8") as train_file:
        train_file.write(train_list)
    
    with open(os.path.join(root_path, "verify.csv"), "w+", encoding="utf-8") as verify_file:
        verify_file.write(verify_list)

def create_test_data_files():
    test_root = r"E:\Data\MLData\视觉特征编码\test_A"
    feature_root = os.path.join(test_root, "test_A_feature")
    test_files = os.listdir(feature_root)
    test_file_list = ""
    for test_f in test_files:
        test_f_path = os.path.join(feature_root, test_f)
        test_file_list = test_file_list + test_f_path + ",\n"
    with open(os.path.join(test_root, "test.csv"), "w+", encoding="utf-8") as train_file:
        train_file.write(test_file_list)

if __name__ == "__main__":
    # load_npy()
    # statistic_label(r"E:\Data\MLData\视觉特征编码\train\train_list.txt")
    # save_npy_as_image(r"E:\Data\MLData\视觉特征编码\train\train_list.txt")
    # split_train_verfy(r"E:\Data\MLData\视觉特征编码\train\train_list.txt")
    create_test_data_files()


