
import numpy as np
import pandas as pd
import torch
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_score
from dataset import get_train_transforms, get_valid_transforms, LeafDataset
from util import MetricMonitor, calculate_f1_macro, accuracy, adjust_learning_rate
from config import params


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

        if i % 10 == 0:
            f1_macro = calculate_f1_macro(output, target)
            acc = accuracy(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update('Accuracy', acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)
        if i % 10 == 0:
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
            target = target.long()
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
    df = params["df"]
    root_path = params["root_path"]
    sub_df = params["sub_df"]
    work_space = params["work_space"]

    train_transforms = get_train_transforms()
    valid_transforms = get_valid_transforms()
    for k, (train_index, test_index) in enumerate(kf.split(df['image'], df['label'])):
        train_img, valid_img = df['image'][train_index], df['image'][test_index]
        train_labels, valid_labels = df['label'][train_index], df['label'][test_index]

        train_paths = root_path + train_img
        valid_paths = root_path + valid_img
        test_paths = root_path + sub_df['image']

        train_dataset = LeafDataset(images_filepaths=train_paths.values,
                                    labels=train_labels.values,
                                    transform=train_transforms)
        valid_dataset = LeafDataset(images_filepaths=valid_paths.values,
                                    labels=valid_labels.values,
                                    transform=valid_transforms)
        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], pin_memory=True,
        )

        val_loader = DataLoader(
            valid_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], pin_memory=True,
        )
        model = LeafNet()
        # model = nn.DataParallel(model)
        model = model.to(params['device'])
        criterion = nn.CrossEntropyLoss().to(params['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for epoch in range(1, params['epochs'] + 1):
            train(train_loader, model, criterion, optimizer, epoch, params)
            acc = validate(val_loader, model, criterion, epoch, params)
            torch.save(model.state_dict(), f"{work_space}\model\leaf_classify\checkpoints\{params['model']}_{k}flod_{epoch}epochs_accuracy{acc:.5f}_weights.pth")


def valid_model():
    df = params["df"]
    sub_df = params["sub_df"]
    label_inv_map = params["label_inv_map"]

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


def valid_model1():
    df = params["df"]
    sub_df = params["sub_df"]
    label_inv_map = params["label_inv_map"]

    # train_img, valid_img = df['image'], df['image']
    test_paths = params["root_path"] + sub_df['image']

    model_name = ['seresnext50_32x4d']
    model_path_list = [
        r'D:\Data\model\leaf_classify\checkpoints\seresnext50_32x4d_0flod_48epochs_accuracy0.97880_weights.pth',
    ]
    model = LeafNet(model_name[0])
    model = model.to(params['device'])
    model.load_state_dict(torch.load(model_path_list[0]))
    model.eval()
        
    test_dataset = LeafDataset(images_filepaths=test_paths,
                                labels=sub_df['image'],
                                transform=get_valid_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=24, shuffle=False,
        num_workers=10, pin_memory=True
    )
    preds = []
    stream = tqdm(test_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            y_hat = model(images)
            y_hat = y_hat.argmax(dim=-1).cpu().numpy().tolist()            
            for oh, name in zip(y_hat, target):
                lbs = label_inv_map[oh]
                preds.append(dict(image=name, labels=lbs))
    df_preds = pd.DataFrame(preds)
    sub_df['label'] = df_preds['labels']
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()

if __name__ == "__main__":
    # train_model()
    valid_model1()
