import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math

import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score

# Metric
from sklearn.metrics import f1_score, accuracy_score

# Augmentation
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


seed = 415

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

root_path = r'E:\Data\MLData\classify\leaves-classify\\'
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


def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ]
    )

class LeafDataset(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        self.images_filepaths = images_filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average='macro')


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
    
def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, params['lr'], params['epochs'], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


""" learning rate schedule """
def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr

params = {
    'model': 'seresnext50_32x4d',
    # 'model': 'resnet50d',
    'device': device,
    'lr': 1e-3,
    'batch_size': 24, # 64
    'num_workers': 0,
    'epochs': 50,
    'out_features': df['label'].nunique(),
    'weight_decay': 1e-5
}


class LeafNet(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['out_features'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x

def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params['device'], non_blocking=True)
        target = target.to(params['device'], non_blocking=True)
        target = target.long()
        output = model(images)
        loss = criterion(output, target)
        f1_macro = calculate_f1_macro(output, target)
        acc = accuracy(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('F1', f1_macro)
        metric_monitor.update('Accuracy', acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"]

def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            f1_macro = calculate_f1_macro(output, target)
            acc = accuracy(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Accuracy']["avg"]

def train_model():
    kf = StratifiedKFold(n_splits=5)

    for k, (train_index, test_index) in enumerate(kf.split(df['image'], df['label'])):
        train_img, valid_img = df['image'][train_index], df['image'][test_index]
        train_labels, valid_labels = df['label'][train_index], df['label'][test_index]

        train_paths = root_path + train_img
        valid_paths = root_path + valid_img
        test_paths = root_path + sub_df['image']

        train_dataset = LeafDataset(images_filepaths=train_paths.values,
                                    labels=train_labels.values,
                                    transform=get_train_transforms())
        valid_dataset = LeafDataset(images_filepaths=valid_paths.values,
                                    labels=valid_labels.values,
                                    transform=get_valid_transforms())
        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], pin_memory=True,
        )

        val_loader = DataLoader(
            valid_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], pin_memory=True,
        )
        model = LeafNet()
        model = nn.DataParallel(model)
        model = model.to(params['device'])
        criterion = nn.CrossEntropyLoss().to(params['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for epoch in range(1, params['epochs'] + 1):
            train(train_loader, model, criterion, optimizer, epoch, params)
            acc = validate(val_loader, model, criterion, epoch, params)
            torch.save(model.state_dict(), f"E:\Data\model\leaf_classify\checkpoints\{params['model']}_{k}flod_{epoch}epochs_accuracy{acc:.5f}_weights.pth")


def valid_model():
    train_img, valid_img = df['image'], df['image']
    train_labels, valid_labels = df['label'], df['label']

    train_paths = '../input/classify-leaves/' + train_img
    valid_paths = '../input/classify-leaves/' + valid_img
    test_paths = '../input/classify-leaves/' + sub_df['image']

    model_name = ['seresnext50_32x4d', 'resnet50d']
    model_path_list = [
        '../input/checkpoints/seresnext50_32x4d_0flod_50epochs_accuracy0.97985_weights.pth',
        '../input/checkpoints/seresnext50_32x4d_1flod_50epochs_accuracy0.97872_weights.pth',
        '../input/checkpoints/seresnext50_32x4d_2flod_36epochs_accuracy0.97710_weights.pth',
        '../input/checkpoints/seresnext50_32x4d_3flod_40epochs_accuracy0.98303_weights.pth',
        '../input/checkpoints/seresnext50_32x4d_4flod_46epochs_accuracy0.97899_weights.pth',
        '../input/checkpoints/resnet50d_0flod_40epochs_accuracy0.98087_weights.pth',
        '../input/checkpoints/resnet50d_1flod_46epochs_accuracy0.97710_weights.pth',
        '../input/checkpoints/resnet50d_2flod_32epochs_accuracy0.97656_weights.pth',
        '../input/checkpoints/resnet50d_3flod_38epochs_accuracy0.97953_weights.pth',
        '../input/checkpoints/resnet50d_4flod_50epochs_accuracy0.97791_weights.pth',
    ]

    model_list = []
    for i in range(len(model_path_list)):
        if i < 5:
            model_list.append(LeafNet(model_name[0]))
        if 5 <= i < 10:
            model_list.append(LeafNet(model_name[1]))
        model_list[i] = nn.DataParallel(model_list[i])
        model_list[i] = model_list[i].to(params['device'])
        init = torch.load(model_path_list[i])
        model_list[i].load_state_dict(init)
        model_list[i].eval()
        model_list[i].cuda()

        
    labels = np.zeros(len(test_paths)) # Fake Labels
    test_dataset = LeafDataset(images_filepaths=test_paths,
                                labels=labels,
                                transform=get_valid_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=10, pin_memory=True
    )


    predicted_labels = []
    pred_string = []
    preds = []

    with torch.no_grad():
        for (images, target) in test_loader:
            images = images.cuda()
            onehots = sum([model(images) for model in model_list]) / len(model_list)
            for oh, name in zip(onehots, target):
                lbs = label_inv_map[torch.argmax(oh).item()]
                preds.append(dict(image=name, labels=lbs))

    df_preds = pd.DataFrame(preds)
    sub_df['label'] = df_preds['labels']
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()

if __name__ == "__main__":
    train_model()

