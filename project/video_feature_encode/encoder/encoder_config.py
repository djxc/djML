import torch

EPOCH = 15
BATCH_SIZE = 64
n = 2   # num_workers
LATENT_CODE_NUM = 32   
log_interval = 10
device='cuda' if torch.cuda.is_available() else"cpu"
