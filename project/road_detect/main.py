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

from model.unet import Unet
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
    model = Unet(3, 1).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(WORKSPACE + args.ckpt))        # 加载训练数据权重
    criterion = nn.BCEWithLogitsLoss()                      # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)      # 优化函数


    liver_dataset = RoadDataset(train_file, train_mode=args.action,
                              transform=x_transforms, target_transform=y_transforms)    
    dataloaders = DataLoader(liver_dataset,
                             batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据
    num_epochs = 100
    dataNum = len(liver_dataset)
    for epoch in range(num_epochs):
        epoch_loss = 0
        step = 0
        with tqdm(total=dataNum, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for x, y, _ in dataloaders:
                step += 1
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
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(x.shape[0])              
        loss_tmp = epoch_loss/step
        print("epoch %d loss:%0.5f" % (epoch, loss_tmp))
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, 'weights_unet_road_{}_{}.pth'.format(epoch, str(loss_tmp)[:7])))        # 保存模型参数，使用时直接加载保存的path文件


def verify(args, model=None):
    import matplotlib.pyplot as plt
    """测试模型，显示模型的输出结果"""
    batch_size = 1 #args.batch_size                    # 每次计算的batch大小
    if model is None:
        model = Unet(3, 1)
        model.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage.cuda(0)))        # 加载训练数据权重
    
    liver_dataset = RoadDataset(verify_file, train_mode=args.action,
                              transform=y_transforms, target_transform=y_transforms)    
    dataloaders = DataLoader(liver_dataset,
                             batch_size=batch_size, shuffle=False, num_workers=1)  # 使用pytorch的数据加载函数加载数据
    model.eval()
    miou_list = []
    dataNum = len(liver_dataset)
    with torch.no_grad():
        with tqdm(total=dataNum, desc=f'verify', unit='img') as pbar:
            for x, y_hat, x_path in dataloaders:
                y = model(x)
                img_y = predb_to_mask(y[0][0])
                result = np.where(img_y > 0.001, 1, 0).astype(np.uint8)           
                miou = calculate_miou(result, y_hat[0][0])
                miou_list.append(miou)
                # result_image = Image.fromarray(result, 'L')
                # result_path = os.path.join(result_folder, "{}_result.png".format(Path(x_path[0]).stem))

                # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
                # ax[0].imshow(torch.squeeze(x).numpy().transpose(1, 2, 0))
                # ax[0].set_title('raw image')
                # ax[1].imshow(y_hat[0][0])
                # ax[1].set_title('true label')
                # ax[2].imshow(result_image)
                # ax[2].set_title('my label')
                # # plt.show()
                # plt.savefig(result_path)
                # plt.close()
                pbar.set_postfix(**{'miou': np.array(miou_list).mean()})
                pbar.update(x.shape[0])            
    return np.array(miou_list).mean()

def calculate_miou(predictions, targets):
    """计算平均交并比"""
    predictions = predictions.reshape(-1)
    targets = targets.float().view(-1).numpy()
    # 计算交集
    intersection = (predictions * targets).sum()
 
    # 计算并集
    total = (predictions + targets).sum() - intersection
 
    # 计算IOU
    iou = (intersection + 1e-10) / (total + 1e-10)
 
    # 计算mIOU
    mIOU = iou.mean()
 
    return mIOU

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
    return torch.functional.F.softmax(predb, 0).cpu()

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
    args.batch_size = 4
    args.ckpt = None # r"D:\Data\MLData\rs_road\model\weights_unet_road_19_0.00229.pth"
    print(args.action)
    # python main.py test --ckpt weight_19.pth#
    if args.action == "train":
        train(args)
    elif args.action == "verify":
        verify(args)
    elif args.action == "test":
        test(args)
