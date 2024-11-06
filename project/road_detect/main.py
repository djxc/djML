"""
使用unet进行目标检测
@author djxc
@date 2019-12-08
"""
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
import PIL.Image as Image
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model.unet import Unet, UNet1
from data.dataset import RoadDataset
from config import WORKSPACE, MODEL_FOLDER, train_file, verify_file, result_folder, test_file, test_result_folder


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
  
def train(args):
    """训练模型  
        1、创建unet模型，并绑定设备
        2、加载数据，使用batch
        3、训练模型，输入
    """
    batch_size = args.batch_size                    # 每次计算的batch大小
    model = UNet1(3, 1).to(device)
    if args.ckpt:
        load_model_ckpt(model, args.ckpt)       
    criterion = nn.BCELoss()                     # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.1)      # 优化函数


    liver_dataset = RoadDataset(train_file, train_mode=args.action,
                              transform=x_transforms, target_transform=y_transforms)    
    dataloaders = DataLoader(liver_dataset,
                             batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据
    
    verify_liver_dataset = RoadDataset(verify_file, train_mode="verify",
                              transform=y_transforms, target_transform=None)    
    verify_dataloaders = DataLoader(verify_liver_dataset,
                             batch_size=1, shuffle=False, num_workers=1)  # 使用pytorch的数据加载函数加载数据
    
    num_epochs = 100
    dataNum = len(liver_dataset)
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        with tqdm(total=dataNum, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for i, (x, y, _) in enumerate(dataloaders):
                # 将输入的要素用gpu计算
                inputs = x.to(device)
                labels = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)   # 损失函数
                loss.backward()                     # 后向传播
                optimizer.step()                    # 参数优化
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': epoch_loss/(i + 1)})
                pbar.update(x.shape[0])    
                if i > 40:
                    break          
        miou = verify(model, verify_dataloaders, criterion)
        save_model(model, epoch, epoch_loss / (i + 1), miou)

def save_model(model, epoch, loss, miou):
    """"""
    model_path = os.path.join(MODEL_FOLDER, 'weights_unet_road_{}_{}_{}.pth'.format(epoch, str(loss)[:7], str(miou.item())[:6]))
    torch.save(model.state_dict(), model_path)        # 保存模型参数，使用时直接加载保存的path文件

def load_model_ckpt(model, ckpt_name):
    """"""
    model_path = os.path.join(MODEL_FOLDER, ckpt_name)
    model.load_state_dict(torch.load(model_path))        # 加载训练数据权重


def verify(model, dataloaders, criterion):
    """测试模型，显示模型的输出结果"""
    model.eval()
    miou_list = []
    dataNum = len(dataloaders)
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=dataNum, desc=f'verify', unit='img') as pbar:
            for i, (x, y_hat, x_path) in enumerate(dataloaders):
                x = x.to(device)
                y_hat = y_hat.to(device)
                y = model(x)    
                # print(y.max(), y.min(), y_hat.max(), y_hat.min())
                y = torch.where(y > 0.1, 1, 0).float()
                
                loss = criterion(y, y_hat) 
                total_loss += loss.item()
                miou = calculate_miou(y, y_hat)
                miou_list.append(miou.item())               
                pbar.set_postfix(**{'miou': np.array(miou_list).mean(), 'loss': total_loss / (i + 1)})
                pbar.update(x.shape[0])     
                if i > 20:
                    break                       
    miou = np.array(miou_list).mean()
    return miou

def verify_test(args):
    """"""
    model = UNet1(3, 1).to(device)
    criterion = nn.BCELoss() 
    load_model_ckpt(model, args.ckpt)

    verify_liver_dataset = RoadDataset(verify_file, train_mode="verify",
                              transform=y_transforms, target_transform=None)    
    verify_dataloaders = DataLoader(verify_liver_dataset,
                             batch_size=1, shuffle=False, num_workers=1)  # 使用pytorch的数据加载函数加载数据
    verify(model, verify_dataloaders, criterion)

def calculate_miou(predictions: torch.tensor, targets: torch.tensor):
    """计算平均交并比"""
    predictions = predictions.reshape(-1)
    targets = targets.view(-1)
    
    # TP表示被正确识别为道路区域的像元数，
    tp = (predictions * targets).sum()
    # FP表示实际非道路但识别为道路的像元数，
    fp = torch.where((predictions - targets) > 0, 1, 0).sum()
    # TN表示被正确识别的非道路区域像元），
    tn = torch.where((predictions + targets) == 0, 1, 0).sum()
    # FN表示实际是道路但未被识别为道路的像元数
    fn = torch.where((targets - predictions) > 0, 1, 0).sum()

    # mIOU=0.5×TP/(TP+FP+FN)+0.5×TN/(TN+FP+FN), 
    mIOU = 0.5 * tp / (tp + fp + fn + 1e-10) + 0.5 * tn / (tn + fp + fn + 1e-10)
 
    return mIOU

def calculate_iou(predictions: torch.tensor, targets: torch.tensor):
    """计算平均交并比
        iou = 相交面积/(预测面积+真实面积-相交面积)
    """
    predictions = predictions.reshape(-1)
    targets = targets.view(-1)
    pred_area = predictions.sum()
    true_area = targets.sum()

    # TP表示被正确识别为道路区域的像元数，
    tp = (predictions * targets).sum()
    iou = tp / ( pred_area + true_area - tp)   
 
    return iou

def test(args):

    """测试模型，显示模型的输出结果"""
    batch_size = 1 #args.batch_size                    # 每次计算的batch大小

    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))        # 加载训练数据权重
    
    liver_dataset = RoadDataset(test_file, train_mode=args.action,
                              transform=y_transforms, target_transform=y_transforms)    
    dataloaders = DataLoader(liver_dataset,
                             batch_size=batch_size, shuffle=False, num_workers=1)  # 使用pytorch的数据加载函数加载数据
    model.eval()

    with torch.no_grad():
        for x, y_hat, x_path in dataloaders:
            y = model(x)
            img_y = predb_to_mask(y[0][0])
            result = np.where(img_y > 0.001, 1, 0).astype(np.uint8)           
            result_image = Image.fromarray(result, 'L')
            result_path = os.path.join(test_result_folder, "{}.png".format(Path(x_path[0]).stem))
            result_image.save(result_path)          
            


def predb_to_mask(predb):
    return torch.functional.F.softmax(predb, 0)

if __name__ == '__main__':
    '''参数解析
        1、首先创建参数解析对象,可用通过对象获取具体的参数
        2、必填参数可以直接输入，可选参数需要`--××`这种格式，在命令行中也是这样设置
    '''
    parse = argparse.ArgumentParser()
    # parse.description = "设置训练还是推理"
    # parse.add_argument("action", type=str, help="train or test", default="train")
    # parse.add_argument("--batch_size", type=int, default=2, required=False)
    # parse.add_argument("--ckpt", type=str,
    #                    help="the path of model pre-train weight file", default=None, required=False)
    args = parse.parse_args()
    args.action = "train"
    args.batch_size = 8
    args.ckpt = None #"weights_unet_road_11_0.22421_nan.pth"
    print(args.action)
    # verify_test(args)
    # python main.py test --ckpt weight_19.pth#
    if args.action == "train":
        train(args)
    elif args.action == "verify":
        verify(args)
    elif args.action == "test":
        test(args)
