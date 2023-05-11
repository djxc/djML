"""
视觉特征编码
@author djxc
@date 2023-05-09
"""
import os
import torch
import argparse
import time
from model import MLPModel, LeNet
from torch import nn, optim
from tqdm import tqdm

from data import VideoFeatureDataset
from torch.utils.data import DataLoader
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 50
lr = 0.001

workspace_root = r"E:\Data\MLData\视觉特征编码"
train_data_file = os.path.join(workspace_root, "train\\train.csv")
verify_data_file = os.path.join(workspace_root, "train\\verify.csv")
model_name = "LeNet"

def train(args):
    """训练模型  
        1、创建unet模型，并绑定设备
        2、加载数据，使用batch
        3、训练模型，输入
    """
    model = LeNet(1, 5).to(device)
    # if args.ckpt:
    #     model.load_state_dict(torch.load(os.path.join(rootPath, args.ckpt)))        # 加载训练数据权重
    batch_size = args.batch_size                    # 每次计算的batch大小
    criterion = nn.CrossEntropyLoss()              # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)      # 优化函数
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_video_feature_dataset = VideoFeatureDataset(train_data_file, mode="train")
    train_data = DataLoader(train_video_feature_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据

    verify_video_feature_dataset = VideoFeatureDataset(verify_data_file, mode="verify")
    verify_data = DataLoader(verify_video_feature_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据

    log_file = open(r"{}\train_log_{}_pre.txt".format(workspace_root, model_name), "a+")
    log_file.write("{} start training……\r\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) 
    best_acc = 0
    for epoch in range(num_epochs):           
        epoch_loss = 0
        model.train()
        startTime = time.time()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        pbar_desc = 'Epoch {0:d}/{1:d}|lr:{2:1.5f}'.format(epoch + 1, num_epochs, current_lr)
        with tqdm(total=len(train_data.dataset), desc=pbar_desc, unit='img') as pbar:
            for step, (x, y) in enumerate(train_data):
                # 将输入的要素用gpu计算
                inputs = x.to(device)
                labels = y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                inputs = inputs.unsqueeze(0)
                inputs = inputs.transpose(0, 1)
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)   # 损失函数
                loss.backward()                     # 后向传播
                optimizer.step()                    # 参数优化
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': '{0:1.5f}'.format(epoch_loss/(step + 1))})
                pbar.update(x.shape[0])     
        
        if epoch > 0 and (epoch + 1) % 2 == 0:
            acc = predictNet(model, verify_data, log_file, batch_size)    
            if acc > best_acc:
                best_acc = acc
                print("save best model, epoch:{}".format(epoch))
                torch.save(model.state_dict(), os.path.join(workspace_root, 'best_model.pth'))        # 保存模型参数，使用时直接加载保存的path文件
        # 每10轮保存一次结果
        if epoch > 0 and (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(workspace_root, 'wight_{}.pth'.format(epoch)))        # 保存模型参数，使用时直接加载保存的path文件
        endTime = time.time()
        log_info = 'epoch %d, loss %.4f, lr %.6f, use time:%.2fs\n' % (epoch + 1, epoch_loss/step, current_lr, endTime - startTime)
        log_file.write(log_info) 
        scheduler.step()



def predictNet(net, test_data, log_file, batchSize):
    ''' 预测 '''
    startTime = time.time()
    accNum = 0
    net.eval()
    for i, (features, labels) in enumerate(test_data):     
        X, Y = features.to(device), labels.to(device)
        X = X.unsqueeze(0)
        X = X.transpose(0, 1)
        y_hat = net(X)
        Y = Y.squeeze(dim=1)
        result = torch.eq(y_hat.max(1, keepdim=True)[1], Y.max(1, keepdim=True)[1])
        accNum = accNum + result.sum().item()
    use_time = time.time() - startTime
    acc = accNum / (len(test_data) * batchSize)
    log_str = "train acc: %.4f, use time:%.2fs" % (acc, use_time)
    print(log_str)
    log_file.write(log_str)
    return acc

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
    parse.add_argument("--action", type=str, default="train", help="train or test, default train")
    parse.add_argument("--batch_size", type=int, default=2, help="batch_size, default 2")
    parse.add_argument("--ckpt", type=str,
                       help="the path of model weight file")
    args = parse.parse_args()
    print(args.action) 
    if args.action == "train":
        train(args)
    elif args.action == "test":
        # python main.py test --ckpt weight_19.pth#
        test(args)
