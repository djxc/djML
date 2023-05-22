
import os
import torch

resume = True
batchSize = 24
learing_rate = 0.01
num_epochs = 300
class_num = 21
model_name = "resNet50_pre"

best_model_name = "best_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_root = r"D:\Data\MLData\classify\UCMerced_LandUse"
model_root = r"{}\model".format(base_root)
if not os.path.exists(model_root):
    os.mkdir(model_root)
train_dataset_file = r"{}\train.txt".format(base_root)
verify_dataset_file = r"{}\verify.txt".format(base_root)