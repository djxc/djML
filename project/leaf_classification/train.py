import os
import time
import datetime
import torch
import torchvision
import numpy as np
import torch.nn as nn
from data import mixup_data, cutmix_data, rotate_data
from model import createResNet, ResNet34
from data import load_leaf_data, load_test_leaf_data, categories
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resume = True
batchSize = 24
 
log = ["{} batchSize: {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batchSize)]

model_name = "resNet"

model_path = r"D:\Data\model\leaf_classify\resnet"

def train():
    '''训练
        1、首先用restNet进行训练, batchSize=6学习率0.01，训练50轮，测试集精度0.88
            restNet训练, batchSize=12学习率0.01，训练200轮，测试集精度0.878
        2、数据增强，TODO
            增加mixup、cutmix以及flip三种增强方式，训练200轮，测试集精度0.94
        3、处理类别不平衡问题，欠采样-EasyEnsemble，从多类中有放回的采取少数类个数，与少数类形成多个数据集，训练多个模型，TODO
        4、增大batch_size会使学习速度下降，但可以提高模型的泛化性
        5、利用训练好的模型进行微调
    '''
    num_epochs = 200
    train_data, test_data = load_leaf_data(batchSize)
    best_acc = 0
    net = create_net(model_name, 176)
    trainer = torch.optim.SGD(net.parameters(), lr=0.005, weight_decay=5e-4)
    net = net.to(device)
    if resume:
        load_net(net)
    # 定义两类损失函数
    loss = torch.nn.CrossEntropyLoss()
    # loss = mixup_criterion
    log_file = open(r"D:\Data\model\train_log_{}.txt".format(model_name), "a+")
    for epoch in range(num_epochs):
        startTime = time.time()
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        net.train()
        loss_total = 0
        for i, (features, labels) in enumerate(train_data):          
            trainer.zero_grad()
            X, Y = features.to(device), labels.to(device)
            # X, Y, y_b, lam = data_augmentation()

            y_hat = net(X)            
            Y = Y.squeeze(dim=1)

            l = loss(y_hat, Y)
            # y_b = y_b.squeeze(dim=1)
            # l = loss(y_hat, Y, y_b, lam)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            loss_total = loss_total + l
        endTime = time.time()
        if epoch > 0 and epoch % 5 == 0:
            acc = predictNet(net, test_data)
            if acc > best_acc:
                best_acc = acc
                save_net(net, "{}\\best_model.pth".format(model_path))
        # 每10轮保存一次结果
        if epoch > 0 and (epoch + 1) % 10 == 0:
            save_net(net, "{}\\{}_epoch.pth".format(model_path, epoch))
        log_info = 'epoch %d, loss %.4f, use time:%.2fs\n' % (epoch + 1, loss_total, endTime - startTime)
        log_file.write(log_info) 
        log.append(log_info)        
        print('epoch %d, loss %.4f, use time:%.2fs' % (
            epoch + 1, loss_total, endTime - startTime))
    log_file.close()

def load_net(net):
    """加载模型的参数"""
    best_model_path = "{}\\best_model.pth".format(model_path)
    if os.path.exists(best_model_path):
        print("加载模型。。。")
        net.load_state_dict(torch.load(best_model_path))


def save_net(net, model_path: str):
    """保存模型"""
    torch.save(net.state_dict(), model_path)

def create_net(net_name: str, class_num: int):
    """根据模型名称创建模型
    """
    if net_name == "efficientNet":
        net = EfficientNet.from_pretrained('efficientnet-b4',  num_classes=class_num)
        # net = EfficientNet.from_name('efficientnet-b4',  num_classes=176)
    elif net_name == "resNet":
        net = createResNet()
        # net = torchvision.models.resnet50(pretrained=True, num_classes=class_num)
    return net


def data_augmentation():
    # 数据增强
    random_num = np.random.random()
    if random_num <= 1/4:
        X, Y, y_b, lam = mixup_data(X, Y, use_cuda=True)
    elif random_num <= 2/4:
        X, Y, y_b, lam = cutmix_data(X, Y, use_cuda=True)
    elif random_num <= 3/4:
        X, Y, y_b, lam = rotate_data(X, Y, use_cuda=True)
    else:
        X, Y, y_b, lam = mixup_data(X, Y, alpha=0, use_cuda=True)
    X, Y, y_b = map(torch.autograd.Variable, (X, Y, y_b))
    return X, Y, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    """损失函数"""
    c = nn.CrossEntropyLoss()    
    return lam * c(pred, y_a) + (1 - lam) * c(pred, y_b)


def predictNet(net, test_data):
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
    print("train acc: %.4f, use time:%.2fs" % (acc, use_time))
    log.append("train acc: %.4f, use time:%.2fs\n" % (acc, use_time))
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

    with open(r"D:\Data\MLData\classify-leaves\result.csv", "w") as result:
        for pred in pred_result:
            result.write("{},{}\n".format(pred["image_name"], pred["label"]))
    


if __name__ == "__main__":
    # net = createResNet()
    # print(net)
    # print(ResNet34())
    # print(torchvision.models.resnet50(pretrained=False, num_classes=176))
    train()
    # verify()