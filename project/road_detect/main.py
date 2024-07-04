"""
使用unet进行目标检测
@author djxc
@date 2019-12-08
"""
import os
import torch
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model.unet import Unet
from data.dataset import RoadDataset
from config import WORKSPACE, MODEL_FOLDER, train_file, verify_file


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


    liver_dataset = RoadDataset(train_file, verify_file, trainOrTest=args.action,
                              transform=x_transforms, target_transform=y_transforms)    
    dataloaders = DataLoader(liver_dataset,
                             batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据
    num_epochs = 20
    dataNum = len(liver_dataset)
    for epoch in range(num_epochs):
        epoch_loss = 0
        step = 0
        with tqdm(total=dataNum, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
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
        loss_tmp = epoch_loss/step
        print("epoch %d loss:%0.5f" % (epoch, loss_tmp))
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER, 'weights_unet_road_{}_{}.pth'.format(epoch, str(loss_tmp)[:7])))        # 保存模型参数，使用时直接加载保存的path文件


def test(args):
    import matplotlib.pyplot as plt
    """测试模型，显示模型的输出结果"""
    batch_size = 1 #args.batch_size                    # 每次计算的batch大小

    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))        # 加载训练数据权重
    
    liver_dataset = RoadDataset(train_file, verify_file, trainOrTest=args.action,
                              transform=y_transforms, target_transform=y_transforms)    
    dataloaders = DataLoader(liver_dataset,
                             batch_size=batch_size, shuffle=True, num_workers=1)  # 使用pytorch的数据加载函数加载数据
    model.eval()

    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            y = torch.functional.F.softmax(y, 0)
            y = y.argmax(0)
            img_y = torch.squeeze(y).numpy()
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
            ax[0].imshow(torch.squeeze(x).numpy().transpose(1, 2, 0))
            ax[0].set_xlabel('Epchos')
            ax[0].set_ylabel('Log(sum squared error)')
            ax[0].set_title('Learning rate 0.01')
            ax[1].imshow(img_y)
            ax[1].set_xlabel('Epchos')
            ax[1].set_ylabel('Log(sum squared error)')
            ax[1].set_title('Learning rate 0.0001')
            plt.show()


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
    args.action = "test"
    args.batch_size = 4
    args.ckpt = r"D:\Data\MLData\rs_road\model\weights_unet_road_19_0.00229.pth"
    if args.action == "train":
        train(args)
        print('train')
    elif args.action == "test":
        # python main.py test --ckpt weight_19.pth#
        test(args)
        print('test')
