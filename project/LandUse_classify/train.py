import os
import time
import datetime
import torch
import torchvision
import numpy as np
import torch.nn as nn
from data import LandUseClassifyDataset, load_land_use_dataset
from config import model_root, device, batchSize, model_name, class_num, resume, learing_rate, num_epochs, best_model_name

from model import LandUseNet



log_header = "{} batchSize: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batchSize)

def train():
    '''训练
        1、首先用restNet进行训练, batchSize=6学习率0.01，训练50轮，测试集精度0.88
            restNet训练, batchSize=12学习率0.01，训练200轮，测试集精度0.878      
    '''
    train_data, test_data = load_land_use_dataset(batchSize)
    best_acc = 0
    landuse_net = LandUseNet(model_name, class_num, resume)
    net = landuse_net.net
    
    trainer = torch.optim.SGD(net.parameters(), lr=learing_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=10, eta_min=0, last_epoch=-1)

    loss = torch.nn.CrossEntropyLoss()    
    try:
        log_file = open(r"{}\train_log_{}_pre.txt".format(model_root, model_name), "a+")
        log_file.write(log_header)
        for epoch in range(num_epochs): 
            startTime = time.time()
            # 训练精确度的和，训练精确度的和中的示例数
            # 绝对误差的和，绝对误差的和中的示例数
            net.train()
            loss_total = 0
            for i, (features, labels) in enumerate(train_data):          
                trainer.zero_grad()
                X, Y = features.to(device), labels.to(device)

                y_hat = net(X)            
                Y = Y.squeeze(dim=1)
                l = loss(y_hat, Y)
                trainer.zero_grad()
                l.backward()

                trainer.step()
                scheduler.step()
                loss_total = loss_total + l
            endTime = time.time()
            if epoch > 0 and epoch % 1 == 0:
                acc = predictNet(net, test_data, log_file)
                if acc > best_acc:
                    best_acc = acc
                    landuse_net.save(best_model_name)
            # 每10轮保存一次结果
            if epoch > 0 and (epoch + 1) % 10 == 0:
                landuse_net.save("{}_epoch.pth".format( epoch))

            log_info = 'epoch %d, loss %.4f, use time:%.2fs, %s\n' % (epoch + 1, loss_total, endTime - startTime, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            log_file.write(log_info) 
            print(log_info)
        log_file.close()
    except Exception as e:
        print(e)
    finally:
        if log_file is not None:
            log_file.close()

def predictNet(net, test_data, log_file):
    '''
        预测
    '''
    startTime = time.time()
    accNum = 0
    net.eval()
    for i, (features, labels) in enumerate(test_data):     
        X, Y = features.to(device), labels.to(device)
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

def verify():
    """"""
    test_data = load_test_leaf_data(8)
    net = create_net(model_name, 176)
    net = net.to(device)
    load_net(net)
    net.eval()
    pred_result = []
    for i, (features, image_names) in enumerate(test_data):     
        X = features.to(device)
        y_hat = net(X)
        y_hat = y_hat.argmax(dim=-1).cpu().numpy().tolist()
        for image_name, y in zip(image_names, y_hat):
            pred_result.append({"image_name": image_name, "label": categories[y]})
        if (i+1) % 100 == 0:
            print("{}/{}".format(i, len(test_data)))

    with open(r"D:\Data\MLData\classify\classify-leaves\result1.csv", "w+") as result:
        for pred in pred_result:
            result.write("{},{}\n".format(pred["image_name"], pred["label"]))
    


if __name__ == "__main__":
    # net = createResNet()
    # print(net)
    # print(ResNet34())
    # print(torchvision.models.resnet50(pretrained=False, num_classes=176))
    train()
    # verify()