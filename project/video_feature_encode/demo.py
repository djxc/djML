import numpy as np
import matplotlib.pyplot as plt
import json

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

if __name__ == "__main__":
    load_npy()
    # statistic_label(r"E:\Data\MLData\视觉特征编码\train\train_list.txt")

