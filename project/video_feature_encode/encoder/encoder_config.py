import torch
workspace_root = r"D:\Data\MLData\mnist"
EPOCH = 15
BATCH_SIZE = 64
n = 2   # num_workers
LATENT_CODE_NUM = 32   # 编码结果长度
log_interval = 50
device='cuda' if torch.cuda.is_available() else"cpu"
