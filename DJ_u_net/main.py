"""
使用unet进行目标检测
@author djxc
@date 2019-12-08
"""
import cv2
import torch
import argparse
from PIL import Image
from unet import Unet
from torch import nn, optim
from dataset import LiverDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import model_path, data_path


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    '''模型训练
        训练模型的细节，需要输入模型、损失函数、优化器以及数据
        1、进行n轮训练，默认为20#
    '''
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)     # 获取数据个数
        epoch_loss = 0
        step = 0
        for x, y in dataload:
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
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))    
    torch.save(model.state_dict(), model_path + 'weightsPed_%d.pth' % epoch)        # 保存模型参数，使用时直接加载保存的path文件
    return model

def train(args):
    """训练模型  
        1、创建unet模型，并绑定设备
        2、加载数据，使用batch
        3、训练模型，输入
    """
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size                    # 每次计算的batch大小
    criterion = nn.BCEWithLogitsLoss()              # 损失函数
    optimizer = optim.Adam(model.parameters())      # 优化函数
    
    liver_dataset = LiverDataset(data_path + "data/peddata",
        transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset,
        batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据
        
    train_model(model, criterion, optimizer, dataloaders)

def test(args):
    """测试模型，显示模型的输出结果"""
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))        # 加载训练数据权重
    liver_dataset = LiverDataset(data_path + "data/pedtest", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
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
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
        print('train')
    elif args.action == "test":
        # python main.py --ckpt weight_19.pth#
        test(args)
        print('test')    