
import torch

resume = True
batchSize = 24
learing_rate = 0.01
num_epochs = 300
class_num = 21
model_name = "resNet50_pre"

best_model_name = "best_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_root = r"E:\Data\MLData\classify\UCMerced_LandUse\model"

train_dataset_file = r"E:\Data\MLData\classify\UCMerced_LandUse\train.txt"
verify_dataset_file = r"E:\Data\MLData\classify\UCMerced_LandUse\verify.txt"