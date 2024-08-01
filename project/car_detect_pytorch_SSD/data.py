
import os
import torch
import random
import torchvision
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as mPolygon
import matplotlib.pyplot as plt
import numpy as np

from util import show_rotate_bboxes
from config import workspace, car_class_list, car_color_list

def split_data():
    """将数据拆分为训练集和验证集8:2
        1、首先将数据打乱，拆分为10份
        2、选取其中2份为验证集，其他的为训练集；然后再选择其他两份为验证集，其他为训练集，依次循环5次
    """
    img_folder = os.path.join(workspace, "input_path")
    label_folder = os.path.join(workspace, "gt")
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
                label_path = os.path.join(label_folder, train_img.replace(".tif", ".xml"))
                img_path_file.write("{},{}\n".format(img_path, label_path))

        with open(os.path.join(workspace, "verify_{}.csv".format(i + 1)), "w") as img_path_file:
            for verify_img in verify_img_list:
                img_path = os.path.join(img_folder, verify_img)
                label_path = os.path.join(label_folder, train_img.replace(".tif", ".xml"))
                img_path_file.write("{},{}\n".format(img_path, label_path))

def cut_data(img_path: str, label_path: str):
    """将图像拆分为256*256的图片
        1、首先将图片拆分为256*256的小图片，如果长宽不是256的倍数则填充为0
        2、将label按照拆分之后的图片范围生成新的label
    """
    save_folder = r"D:\Data\MLData\车辆检测\car_det_train_small\input_path"
    xml_save_folder = r"D:\Data\MLData\车辆检测\car_det_train_small\gt"

    objects_list = read_objects_from_xml(label_path)
    # 裁剪图像，首先裁剪完整的，剩下边角需要用0填充
    img_size = 256
    image_name = Path(img_path).stem
    feature = Image.open(img_path)
    img_width, img_height = feature.width, feature.height
    width_size = img_width // img_size
    height_size = img_height // img_size
    for i in range(width_size + 1):
        for j in range(height_size + 1):
            if i == width_size or j == height_size:               
                if (i == width_size and (i + 1) * img_size > img_width) and (j == height_size and (j + 1) * img_size > img_height):
                    box = (i * img_size, j * img_size, img_width, img_height)
                elif i == width_size and (i + 1) * img_size > img_width:
                    box = (i * img_size, j * img_size, img_width, (j + 1) * img_size)
                elif j == height_size and (j + 1) * img_size > img_height:
                    box = (i * img_size, j * img_size, (i + 1) * img_size, img_height)
                new_img = feature.crop(box)
                padded_img = Image.new(new_img.mode, (img_size, img_size), 0)
                padded_img.paste(new_img, (0, 0), mask=new_img)
                new_img = padded_img
            else:
                box = (i * img_size, j * img_size, (i + 1) * img_size, (j + 1) * img_size)
                new_img = feature.crop(box)
            label_result = filter_label(objects_list, box)
            with open(os.path.join(xml_save_folder, "{}_{}_{}.csv".format(image_name, i, j)), "w") as label_file:
                label_file.writelines(label_result)
            new_img.save(os.path.join(save_folder, "{}_{}_{}.png".format(image_name, i, j)))

def filter_label(objects_list, box):
    """根据当前范围裁剪objects
        1、仅保留再当前范围box内的object，可能会裁剪部分object
        2、将object坐标转为为新图像坐标
        3、返回str每行为：class, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y
    """
    result = []
    for obj in objects_list:
        # 如果对象不在box内,则略过，如果对象完全在box内则记录该对象四个顶点；如果object与box部分相交则需要进行裁剪
        in_point_num = obj_point_in_box(obj, box)
        if in_point_num == 0:
            continue
        elif in_point_num == 4:
            result.append("{},{},{},{},{},{},{},{},{}\n".format(obj[0], obj[1][0] - box[0], obj[1][1] - box[1], obj[2][0] - box[0], obj[2][1] - box[1], obj[3][0] - box[0], obj[3][1] - box[1], obj[4][0] - box[0], obj[4][1] - box[1]))
        else:
            box_polygon = Polygon([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])])
            obj_polygon = Polygon([(obj[1][0], obj[1][1]), (obj[2][0], obj[2][1]), (obj[3][0], obj[3][1]), (obj[4][0], obj[4][1])])            
            inter_polygon = box_polygon.intersection(obj_polygon)
            coords = inter_polygon.exterior.coords
            result.append("{},{},{},{},{},{},{},{},{}\n".format(obj[0], coords[0][0] - box[0], coords[0][1] - box[1], coords[1][0] - box[0], coords[1][1] - box[1], coords[2][0] - box[0], coords[2][1] - box[1], coords[3][0] - box[0], coords[3][1] - box[1]))
    return result

def cut_label(object, box):
    """用box裁剪object
        1、如果object顶点y坐标都在box y坐标范围内，则需要用box竖边进行裁剪
        2、如果object顶点x坐标都在box x坐标范围内，则需要用box横边进行裁剪
        3、如果object顶点xy坐标都不在box xy坐标范围内，则需要用顶点进行裁剪
    """

def obj_point_in_box(obj, box):
    """判断对象有几个角在box内
        1、如果没有一个角在box内则该object不在box内
        2、如果四个顶点在box内则object完全在box内
        3、如果有大于零个点小于4个点在box内则需要将其裁剪
    """
    in_point_num = 0
    for i, info in enumerate(obj):
        if i > 0:
            if is_box_contain_point(box, info):
                in_point_num = in_point_num + 1
    return in_point_num

def is_box_contain_point(box, point):
    """判断box是否包含点
        1、判断点x是否介于box最大x与最小x之间
        2、判断点y是否介于box最大y与最小y之间
    """
    return point[0] >= box[0] and point[0] <= box[2] and point[1] >= box[1] and point[1] <= box[3]
        

def read_objects_from_xml(xml_path: str):
    """从xml中读取目标对象
        @return 目标对象列表，[class, [point1_x, point1_y], [point2_x, point2_y], [point3_x, point3_y], [point4_x, point4_y]]
    """
    label_info = ET.parse(xml_path)  # 替换成你的XML文件路径
    root = label_info.getroot()
    size_info = root.find('size')
    width_size = int(size_info.find("width").text)
    height_size = int(size_info.find("height").text)
    object_list = root.find('objects').findall("object")
    label_list = []
    for object in object_list:
        label_tmp = []
        car_class = object.find("possibleresult").find("name").text 
        label_tmp.append(car_class_list.index(car_class))
        points = object.find("points")
        for i, point in enumerate(points):
            if i < 4:
                point_txt = point.text.split(",")                
                label_tmp.append([float(point_txt[0]), float(point_txt[1])])
        label_list.append(label_tmp)
    return label_list

def read_objects_from_csv(csv_path: str):
    """"""
    vertices_list = []
    labels = []
    with open(csv_path) as csv_file:
        points = csv_file.readlines()
        for point in points:
            vertices = point.replace("\n", "").split(",")
            vertices = [float(v) for v in vertices]
            labels.append(int(vertices[0]))
            vertices_list.append([(vertices[1], vertices[2]), (vertices[3], vertices[4]), (vertices[5], vertices[6]), (vertices[7], vertices[8])]) 
    return vertices_list, labels

def show_img_label(img_path: str, label_path: str):
    """显示图片以及label"""
    vertices_list, labels = read_objects_from_csv(label_path)
    print("find object {}".format(len(vertices_list)))
    # 创建一个图像
    fig, ax = plt.subplots()
    feature = Image.open(img_path)
    ax.imshow(feature)
    show_rotate_bboxes(ax, vertices_list, labels)

    # 显示图像
    plt.show()

def anasys_cls():
    """分析类的分布情况"""
    xml_folder = r"D:\Data\MLData\车辆检测\car_det_train\gt"
    xml_list = os.listdir(xml_folder)
    cls_num = {}
    area_num = {}
    for label_path in xml_list:
        objects_list = read_objects_from_xml(os.path.join(xml_folder, label_path))
        for obj in objects_list:
            obj_polygon = Polygon([(obj[1][0], obj[1][1]), (obj[2][0], obj[2][1]), (obj[3][0], obj[3][1]), (obj[4][0], obj[4][1])])            
            area = obj_polygon.area
            if obj[0] not in cls_num:
                cls_num[obj[0]] = 1
                area_num[obj[0]] = [area]
            else:
                cls_num[obj[0]] = cls_num[obj[0]] + 1
                area_num[obj[0]].append(area)
    for i in cls_num:
        mean_area = np.array(area_num[i]).mean()
        print(i, cls_num[i], mean_area)

class CarDataset(torch.utils.data.Dataset):
    """一个用于加载汽车检测数据集的自定义数据集。"""
    def __init__(self, is_train, workspace, fold_num=1):
        self.is_train = is_train
        self.fold_num = fold_num
        self.image_paths, self.label_paths = self.read_data_car(workspace)     
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(10000),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )       
        print('read ' + str(len(self.image_paths)) + (
            f' training examples' if is_train else f' validation examples'))

    def read_data_car(self, workspace: str):
        """读取汽车检测数据集中的图像和标签路径。"""
        image_paths = []
        label_paths = []
        csv_fname = os.path.join(workspace, 'train_{}.csv'.format(self.fold_num) if self.is_train else 'verify_{}.csv'.format(self.fold_num))
        with open(csv_fname) as csv_file:
            car_data_list = csv_file.readlines()
            for car_data in car_data_list:
                img_label = car_data.replace("\n", "").split(",")
                image_paths.append(img_label[0])                
                label_paths.append(img_label[1])
        return image_paths, label_paths
        # csv_data = pd.read_csv(csv_fname)
        # csv_data = csv_data.set_index('img_name')
        # images, targets = [], []
        # for img_name, target in csv_data.iterrows():
        #     images.append(
        #         torchvision.io.read_image(
        #             os.path.join(data_dir,
        #                         'bananas_train' if is_train else 'bananas_val',
        #                         'images', f'{img_name}')))
        #     # Here `target` contains (class, upper-left x, upper-left y,
        #     # lower-right x, lower-right y), where all the images have the same
        #     # banana class (index 0)
        #     targets.append(list(target))
        # return images, torch.tensor(targets).unsqueeze(1) / 256

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        feature = Image.open(img_path)
        feature = feature.convert('RGB')

        label_info = ET.parse(label_path)  # 替换成你的XML文件路径
        root = label_info.getroot()
        size_info = root.find('size')
        width_size = int(size_info.find("width").text)
        height_size = int(size_info.find("height").text)
        object_list = root.find('objects').findall("object")
        label_list = []
        for object in object_list:
            label_tmp = []
            car_class = object.find("possibleresult").find("name").text 
            label_tmp.append(car_class_list.index(car_class))
            points = object.find("points")
            for i, point in enumerate(points):
                if i == 0 or i == 3:
                    point_txt = point.text.split(",")
                    width = float(point_txt[0])/width_size
                    height = float(point_txt[1])/height_size
                    label_tmp.append(width)
                    label_tmp.append(height)
            label_list.append(label_tmp)
        return (self.transform(feature), torch.tensor(label_list))

    def __len__(self):
        return len(self.image_paths)
    
if __name__ == "__main__":
    # split_data()
    # cut_data(r"D:\Data\MLData\车辆检测\car_det_train\input_path\1.tif", r"D:\Data\MLData\车辆检测\car_det_train\gt\1.xml")
    # show_img_label(r"D:\Data\MLData\车辆检测\car_det_train_small\input_path\1_1_3.png", r"D:\Data\MLData\车辆检测\car_det_train_small\gt\1_1_3.csv")
    anasys_cls()

