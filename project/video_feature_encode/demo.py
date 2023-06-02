import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import random
from PIL import Image
from config import workspace_root

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
    test_root = os.path.join(workspace_root, "test_A")
    feature_root = os.path.join(test_root, "test_A_feature")
    test_files = os.listdir(feature_root)
    test_file_list = ""
    for test_f in test_files:
        test_f_path = os.path.join(feature_root, test_f)
        test_file_list = test_file_list + test_f_path + ",\n"
    with open(os.path.join(test_root, "test.csv"), "w+", encoding="utf-8") as train_file:
        train_file.write(test_file_list)

def k_folder(train_path):
    """这里为3折交叉验证"""
    verify1_list = []
    verify2_list = []
    with open(train_path) as label_file:
        train_data_list = label_file.readlines()
        for train_data in train_data_list:
            if random.random() > 0.5:
                verify1_list.append(train_data)
            else:
                verify2_list.append(train_data)

    print(len(verify1_list), len(verify2_list))
    # np.random.shuffle(label_list)
    with open(os.path.join(workspace_root, "verify1.csv"), "w+", encoding="utf-8") as verify_file:
        verify_file.writelines(verify1_list)

    with open(os.path.join(workspace_root, "verify2.csv"), "w+", encoding="utf-8") as verify_file:
        verify_file.writelines(verify2_list)


def split_train_part(label_path, part_num=5):
    """把数据分为5份"""
    root_path = Path(label_path).parent
    data_root = os.path.join(root_path, "train_feature")
    label_list = []
    with open(label_path) as label_file:
        labels = json.loads(label_file.read())
        for i, label in enumerate(labels):
            cls = labels[label]
            data_file = os.path.join(data_root, label)
            label_list.append("{}, {},\n".format(data_file, cls))
    np.random.shuffle(label_list)
    label_size = len(label_list)
    part_size = int(label_size / part_num)
    part_data_list = []
    for i in range(part_num):
        if i == (part_num - 1):
            part_data = label_list[i * part_size:]
        else:
            part_data = label_list[i * part_size: (i + 1) * part_size]
        part_data_list.append(part_data)
    
    for i in range(part_num):
        verify_list = part_data_list[i]
        train_list = []
        for j in range(part_num):
            if i != j:
                train_list.extend(part_data_list[j])            
        with open(os.path.join(root_path, "train_{}.csv".format(i)), "w+", encoding="utf-8") as train_file:
            train_file.writelines(train_list)  
        with open(os.path.join(root_path, "verify_{}.csv".format(i)), "w+", encoding="utf-8") as verify_file:
            verify_file.writelines(verify_list)   

def max_frequency(result0, result1, result2, result3):
    result_map = {}
    for k in result0:
        r0 = np.array(result0[k]).argmax()
        r1 = np.array(result1[k]).argmax()
        r2 = np.array(result2[k]).argmax()
        r3 = np.array(result3[k]).argmax()
        cls_array = np.array([r0, r1, r2, r3])
        # 找频率最高的
        # 按照每个模型的准确率赋予权重
        unique_arr = np.unique(cls_array)
        if len(unique_arr) == 4:
            print(r0, r1, r2, r3)
            result = r0
        else:           
            # 计算数组中每个数值出现的次数
            counts = np.bincount(cls_array)
            # 找到出现次数最多的数值
            result = np.argmax(counts)
        result_map[k] = str(result)
    return result_map

def sum_probabilty(result0, result1, result2, result3):
    result_map = {}
    for k in result0:
        r0 = np.array(result0[k])
        r1 = np.array(result1[k])
        r2 = np.array(result2[k])
        r3 = np.array(result3[k])
        cls_array = r0 + r1 + r2 + r3
        result = cls_array.argmax()     
        result_map[k] = str(result)
    return result_map

def weight_probabilty(result0, result1, result2, result3):
    weights = [
        [0.722, 0.884, 0.870, 0.939, 0.775],
        [0.889, 0.842, 0.872, 0.800, 0.757],
        [0.652, 0.816, 0.824, 0.771, 0.676],
        [0.714, 0.786, 0.684, 0.943, 0.75]
    ]
    result_map = {}
    for k in result0:
        r0 = np.array(result0[k]) * np.array(weights[0])
        r1 = np.array(result1[k]) * np.array(weights[1])
        r2 = np.array(result2[k]) * np.array(weights[2])
        r3 = np.array(result3[k]) * np.array(weights[3])
        cls_array = r0 + r1 + r2 + r3
        result = cls_array.argmax()     
        result_map[k] = str(result)
    return result_map

def merge_result():
    with open(r"E:\Data\MLData\videoFeature\result_0.txt") as result0_f:
        result0 = json.loads(result0_f.read())

    with open(r"E:\Data\MLData\videoFeature\result_1.txt") as result1_f:
        result1 = json.loads(result1_f.read())

    with open(r"E:\Data\MLData\videoFeature\result_2.txt") as result2_f:
        result2 = json.loads(result2_f.read())

    with open(r"E:\Data\MLData\videoFeature\result_3.txt") as result3_f:
        result3 = json.loads(result3_f.read())

    result_map1 = max_frequency(result0, result1, result2, result3)
    result_map2 = sum_probabilty(result0, result1, result2, result3)
    result_map3 = weight_probabilty(result0, result1, result2, result3)
    count = 0
    for i, item in enumerate(result_map1):
        if result_map1[item] != result_map2[item]:
            print(item)
            count = count + 1
    print("not eq count :{}!!!!!".format(count))
    
    with open(os.path.join(workspace_root, "result.txt"), "w+") as result_f:
        result_f.write(json.dumps(result_map3))
    



if __name__ == "__main__":
    # load_npy()
    # statistic_label(r"E:\Data\MLData\视觉特征编码\train\train_list.txt")
    # save_npy_as_image(r"E:\Data\MLData\视觉特征编码\train\train_list.txt")
    # split_train_verfy(os.path.join(workspace_root, "train", "train_list.txt"))
    # create_test_data_files()
    # k_folder(r"E:\Data\MLData\videoFeature\train\train.csv")
    # split_train_part(r"E:\Data\MLData\videoFeature\train\train_list.txt")
    merge_result()


