
from data.split_data import split_data, create_test_path_file
from config import train_folder

def split_data_test():
    split_data(train_folder)

if __name__ == "__main__":
    split_data_test()
    # create_test_path_file(r"D:\Data\MLData\rs_road\算法赛道1高分辨率遥感数据道路提取初赛数据集\数据集\val\img")