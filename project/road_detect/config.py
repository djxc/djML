import os
WORKSPACE =  r"D:\Data\MLData\road_dataset\CHN6-CUG"
MODEL_FOLDER = os.path.join(WORKSPACE, "model")

train_folder = os.path.join(WORKSPACE, "train")
result_folder = os.path.join(WORKSPACE, "result")

train_file = os.path.join(train_folder, "train_1.csv")
verify_file = os.path.join(train_folder, "verify_1.csv")
test_file = os.path.join(WORKSPACE, "val", "test.csv")
test_result_folder = os.path.join(WORKSPACE, "val", "result")

if not os.path.exists(test_result_folder):
    os.mkdir(test_result_folder)

