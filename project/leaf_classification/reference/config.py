
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

work_space = r"D:\Data"
seed = 415

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

root_path = r'{}\MLData\classify\leaves-classify\\'.format(work_space)
labels_file_path = os.path.join(root_path, 'train.csv')
sample_submission_path = os.path.join(root_path, 'test.csv')

df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
labels_unique = df['label'].unique()


le = LabelEncoder()
le.fit(df['label'])
df['label'] = le.transform(df['label'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
label_inv_map = {v: k for k, v in label_map.items()}


params = {
    'model': 'seresnext50_32x4d',
    # 'model': 'resnet50d',
    'device': device,
    'lr': 1e-3,
    'batch_size': 16, # 64
    'num_workers': 0,
    'epochs': 50,
    'out_features': df['label'].nunique(),
    'weight_decay': 1e-5,
    'label_inv_map': label_inv_map,
    "df": df,
    "sub_df": sub_df,
    "root_path": root_path,
    "work_space": work_space,
}
