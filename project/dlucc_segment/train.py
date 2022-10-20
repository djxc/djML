"""
使用unet进行目标检测
@author djxc
@date 2019-12-08
"""
import os
import torch
import argparse
from PIL import Image
from model import Unet
from torch import nn, optim
from tqdm import tqdm

from data import LUCCDataset, rootPath
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 20

def train(args):
    """训练模型  
        1、创建unet模型，并绑定设备
        2、加载数据，使用batch
        3、训练模型，输入
    """
    model = Unet(3, 1).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(os.path.join(rootPath, args.ckpt)))        # 加载训练数据权重
    batch_size = args.batch_size                    # 每次计算的batch大小
    criterion = nn.BCEWithLogitsLoss()              # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.1)      # 优化函数

    liver_dataset = LUCCDataset("train.csv", mode=args.action)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据

    for epoch in range(num_epochs):
        dt_size = len(dataloaders.dataset)     # 获取数据个数
        epoch_loss = 0
        step = 0
        with tqdm(total=len(dataloaders.dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for x, y in dataloaders:
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
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), os.path.join(rootPath, 'weights_unet_car_%d.pth' % epoch))        # 保存模型参数，使用时直接加载保存的path文件


def test(args):
    import matplotlib.pyplot as plt
    """测试模型，显示模型的输出结果"""
    model = Unet(3, 1)
    model.load_state_dict(torch.load(model_path +
                                     args.ckpt, map_location='cpu'))        # 加载训练数据权重
    liver_dataset = LUCCDataset("train.csv", trainOrTest=args.action)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()

    plt.ion()
    with torch.no_grad():
        dj = 0
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            # x, y = img_y.shape
            # for i in range(x):
            #     for j in range(y):
            #         if img_y[i, j] > 0:
            #             img_y[i, j] = 255
            #         else:
            #             img_y[i, j] = 0
            # im = Image.fromarray(img_y)
            # im.show()
            # if im.mode != 'RGB':
            #     im = im.convert('RGB')
            # im.save("predict_%3d.png"%dj)
            # dj = dj + 1
            plt.imshow(img_y)
            plt.pause(5)
        plt.show()


if __name__ == '__main__':
    '''参数解析
        1、首先创建参数解析对象,可用通过对象获取具体的参数
        2、必填参数可以直接输入，可选参数需要`--××`这种格式，在命令行中也是这样设置
    '''
    parse = argparse.ArgumentParser()
    parse.description = "设置训练还是推理"
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckpt", type=str,
                       help="the path of model weight file")
    args = parse.parse_args()
    print(args.action) 
    if args.action == "train":
        train(args)
    elif args.action == "test":
        # python main.py test --ckpt weight_19.pth#
        test(args)
