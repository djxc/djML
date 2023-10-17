
import torch 
from tqdm import tqdm
from torch import nn, optim
import torch.utils.data.dataloader as dataloader

from model import Unet
from data import CloudDataset

model_path = r"D:\Data\MLData\38cloud"
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cloud_data(data_root, batch_size):
    ''' 加载ITCVD数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = dataloader.DataLoader(
        CloudDataset(True, data_root), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)

    # test_iter = dataloader.DataLoader(
    #     CloudDataset(False, "/2020/clothes_person_test/"), batch_size, drop_last=True,
    #     num_workers=num_workers)
    return train_iter

def train(num_epochs, batch_size, DATA_ROOT):
    """训练模型  
        1、创建unet模型，并绑定设备
        2、加载数据，使用batch
        3、训练模型，输入
    """
    model = Unet(4, 1).to(device)
    # if args.ckpt:
    #     model.load_state_dict(torch.load(
    #         model_path + args.ckpt))        # 加载训练数据权重
    criterion = nn.BCEWithLogitsLoss()              # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)      # 优化函数

    
    train_iter = load_cloud_data(DATA_ROOT, batch_size)  
    dataNum = len(train_iter.dataset)
    '''模型训练
        训练模型的细节，需要输入模型、损失函数、优化器以及数据
        1、进行n轮训练，默认为20#
    '''
    for epoch in range(num_epochs):
        epoch_loss = 0
        step = 0
        with tqdm(total=dataNum, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for x, y in train_iter:
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
                pbar.set_postfix(**{'loss': epoch_loss/step})
                pbar.update(x.shape[0])              
        torch.save(model.state_dict(), model_path + 'weights_unet_car_%d.pth' % epoch)        # 保存模型参数，使用时直接加载保存的path文件
    return model

    
if __name__ == "__main__":
    DATA_ROOT = r"D:\Data\MLData\38cloud"
    train(60, 4, DATA_ROOT) 