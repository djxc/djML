import os
WORKSPACE =  r"D:\Data\MLData\rs_road"

data_folder = os.path.join(WORKSPACE, "算法赛道1高分辨率遥感数据道路提取初赛数据集", "数据集")
train_folder = os.path.join(data_folder, "train")
result_folder = os.path.join(WORKSPACE, "result")

MODEL_FOLDER = os.path.join(WORKSPACE, "model")
train_file = os.path.join(train_folder, "train_1.csv")
verify_file = os.path.join(train_folder, "verify_1.csv")
test_file = os.path.join(data_folder, "val", "test.csv")
test_result_folder = os.path.join(data_folder, "val", "result")

if not os.path.exists(test_result_folder):
    os.mkdir(test_result_folder)

