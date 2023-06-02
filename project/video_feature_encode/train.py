"""
视觉特征编码
@author djxc
@date 2023-05-09
"""
import os
import torch
import argparse
import time
import json
from pathlib import Path
from model import MLPModel, LeNet, AlexNet
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import random

from config import workspace_root, loss_type, lr, class_num, num_epochs, num_workers, model_name, data_part, weight_decay, resume_model
from data import VideoFeatureDataset
from model import MLPModel, LeNet, create_net
from dloss import MultiClassFocalLossWithAlpha

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data_file = os.path.join(workspace_root, "train\\train_{}.csv".format(data_part))
verify_data_file = os.path.join(workspace_root, "train\\verify_{}.csv".format(data_part))
test_data_file = os.path.join(workspace_root, "test_A\\test.csv")

def train(args):
    """训练模型  
        1、创建unet模型，并绑定设备
        2、加载数据，使用batch
        3、训练模型，输入
    """
    log_file = open(r"{}\train_log_{}_{}.txt".format(workspace_root, model_name, data_part), "a+")
    log_file.write("{} start training……\r\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) 
    
    batch_size = args.batch_size                    # 每次计算的batch大小

    train_video_feature_dataset = VideoFeatureDataset(train_data_file, mode="train")
    train_data = DataLoader(train_video_feature_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 使用pytorch的数据加载函数加载数据

    verify_video_feature_dataset = VideoFeatureDataset(verify_data_file, mode="verify")
    verify_data = DataLoader(verify_video_feature_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 使用pytorch的数据加载函数加载数据

    model = create_net(model_name, class_num, args.resume).to(device)
    criterion = create_loss(loss_type)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)      # 优化函数
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    # predictNet(model, verify_data, log_file, batch_size)
    best_acc = 0
    for epoch in range(num_epochs):           
        epoch_loss = 0
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
                if random.random() > 0.5:
                    criterion = mixup_criterion
                    inputs, labels, y_b, lam = data_augmentation(inputs, labels)
                    outputs = model(inputs)               
                    loss = criterion(outputs, labels, y_b, lam)             
                else:
                    criterion = create_loss(loss_type)
                    outputs = model(inputs)                            
                    loss = criterion(outputs, labels)                          

                loss.backward()                     # 后向传播
                optimizer.step()                    # 参数优化
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': '{0:1.5f}'.format(epoch_loss/(step + 1))})
                pbar.update(x.shape[0])     
        
        if epoch > 0 and (epoch + 1) % 2 == 0:
            acc = predictNet(model, verify_data, log_file, batch_size)    
            if acc > best_acc:
                best_acc = acc
                print("save best model, epoch:{}".format(epoch + 1))
                torch.save(model.state_dict(), os.path.join(workspace_root, 'best_model_{}_{}.pth'.format(model_name, data_part)))        # 保存模型参数，使用时直接加载保存的path文件
            model.train()            
        # 每10轮保存一次结果
        if epoch > 0 and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(workspace_root, 'wight_{}_{}_{}.pth'.format(model_name, data_part, epoch + 1)))        # 保存模型参数，使用时直接加载保存的path文件
        endTime = time.time()
        log_info = 'epoch %d, loss %.4f, lr %.6f, use time:%.2fs\n' % (epoch + 1, epoch_loss/step, current_lr, endTime - startTime)
        log_file.write(log_info) 
        log_file.flush()
        scheduler.step()


def create_loss(loss_type):
    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss()  
    else:
        criterion = MultiClassFocalLossWithAlpha([0.4, 0.1, 0.1, 0.1, 0.3])
    return criterion

def mixup_criterion(pred, y_a, y_b, lam):
    """损失函数"""
    # c = nn.CrossEntropyLoss()    
    c = create_loss(loss_type)
    return lam * c(pred, y_a) + (1 - lam) * c(pred, y_b)

def data_augmentation(X, Y):
    # 数据增强       
    X, Y, y_b, lam = mixup_data(X, Y)
    X, Y, y_b = map(torch.autograd.Variable, (X, Y, y_b))
    return X, Y, y_b, lam
    
# 图像增强
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """ Mixup 数据增强 -> 随机叠加两张图像 """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # β分布
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if use_cuda:
        index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam   


def predictNet(net, test_data, log_file, batchSize):
    ''' 预测 '''
    startTime = time.time()
    accNum = 0
    verify_result = {
        "0": {"total": 0, "error": 0, "error_info": {}},
        "1": {"total": 0, "error": 0, "error_info": {}},
        "2": {"total": 0, "error": 0, "error_info": {}},
        "3": {"total": 0, "error": 0, "error_info": {}},
        "4": {"total": 0, "error": 0, "error_info": {}}
        }
    criterion = create_loss(loss_type)
    total_loss = 0
    net.eval() 
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_data):     
            X, Y = features.to(device), labels.to(device)
            # X = X.unsqueeze(0)
            # X = X.transpose(0, 1)
            y_hat = net(X)
            loss = criterion(y_hat, Y)
            total_loss += loss.item()
            Y = Y.squeeze(dim=1)
            if loss_type == "CE":
                y_hat = y_hat.max(1, keepdim=True)[1]
                Y = Y.max(1, keepdim=True)[1]
            else:
                y_hat = y_hat.max(1, keepdim=True).indices.squeeze(dim=1)
            result = torch.eq(y_hat, Y)
            accNum = accNum + result.sum().item()
            result = result.cpu().numpy()
            Y = Y.cpu().numpy()

            if loss_type == "CE":
                for j, r in enumerate(result):
                    y = str(Y[j][0])
                    verify_result[y]["total"] = verify_result[y]["total"] + 1
                    if not r[0]:
                        verify_result[y]["error"] = verify_result[y]["error"] + 1
                        y_pre = str(y_hat[j].cpu().item())
                        if y_pre not in verify_result[y]["error_info"]:
                            verify_result[y]["error_info"][y_pre] = 1
                        else:
                            verify_result[y]["error_info"][y_pre] = verify_result[y]["error_info"][y_pre] + 1
            else:
                for j, r in enumerate(result):
                    y = str(Y[j])
                    verify_result[y]["total"] = verify_result[y]["total"] + 1
                    if not r:
                        verify_result[y]["error"] = verify_result[y]["error"] + 1
                        y_pre = str(y_hat[j].cpu().item())
                        if y_pre not in verify_result[y]["error_info"]:
                            verify_result[y]["error_info"][y_pre] = 1
                        else:
                            verify_result[y]["error_info"][y_pre] = verify_result[y]["error_info"][y_pre] + 1

        use_time = time.time() - startTime
        acc = accNum / (len(test_data) * batchSize)
        log_str = "train acc: %.4f, loss: %.4f; use time:%.2fs\n" % (acc, total_loss/(i + 1), use_time)
        print(log_str)
        for cls in verify_result:
            total_num = verify_result[cls]["total"]
            true_num = total_num - verify_result[cls]["error"]
            error_info_str = json.dumps(verify_result[cls]["error_info"])
            class_result = "cls {0:s} acc is {1:1.3f}, total: {2:d}, error: {3:d}".format(cls, 
                    true_num/total_num, 
                    total_num, verify_result[cls]["error"])
            class_result = "{}, error_info: {}".format(class_result, error_info_str)
            log_file.write(class_result + "\n")
            print(class_result)        
        log_file.write(log_str)
        return acc

def test(net):
    ''' 测试 '''
    test_video_feature_dataset = VideoFeatureDataset(test_data_file, mode="test")
    test_data = DataLoader(test_video_feature_dataset, batch_size=1, num_workers=4)  # 使用pytorch的数据加载函数加载数据
    net.eval()
    result = {}
    with tqdm(total=len(test_data.dataset), desc="test", unit='img') as pbar:
        for i, (features, image_path) in enumerate(test_data):     
            X = features.to(device)
            # X = X.unsqueeze(0)
            # X = X.transpose(0, 1)
            y_hat = net(X)
            result_cls = y_hat.max(1, keepdim=True)[1]      
            image_name = Path(image_path[0]).name
            result[image_name] = str(result_cls.item())
            pbar.update(features.shape[0])   
        print(result)
        with open(os.path.join(workspace_root, "result.txt"), "w+") as result_f:
            result_f.write(json.dumps(result))

def test_kf(net, k_num):
    ''' 测试 '''
    test_video_feature_dataset = VideoFeatureDataset(test_data_file, mode="test")
    test_data = DataLoader(test_video_feature_dataset, batch_size=1, num_workers=4)  # 使用pytorch的数据加载函数加载数据
    net.eval()
    result = {}
    with tqdm(total=len(test_data.dataset), desc="test", unit='img') as pbar:
        for i, (features, image_path) in enumerate(test_data):     
            X = features.to(device)          
            y_hat = net(X)
            result_array = y_hat.squeeze(0).detach().numpy().tolist()
            image_name = Path(image_path[0]).name
            result[image_name] = result_array

            pbar.update(features.shape[0])   
        print(result)
        with open(os.path.join(workspace_root, "result_{}.txt".format(k_num)), "w+") as result_f:
            result_f.write(json.dumps(result))


if __name__ == '__main__':
    '''参数解析
        1、首先创建参数解析对象,可用通过对象获取具体的参数
        2、必填参数可以直接输入，可选参数需要`--××`这种格式，在命令行中也是这样设置
    '''
    parse = argparse.ArgumentParser()
    parse.description = "设置训练还是推理"
    parse.add_argument("--action", type=str, default="train", help="train or test, default train")
    parse.add_argument("--batch_size", type=int, default=2, help="batch_size, default 2")
    parse.add_argument("--resume", type=str,
                       help="resume")
    args = parse.parse_args()
    print(args.action) 
    if args.action == "train":
        train(args)
    elif args.action == "test":
        # python main.py test --ckpt weight_19.pth#
        model = create_net(model_name, class_num, args.resume).to(device)
        print("load {} ...".format(resume_model))
        model.load_state_dict(torch.load(os.path.join(workspace_root, resume_model)))        # 加载训练数据权重
        # test(model)
        test_kf(model, data_part)
