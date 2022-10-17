import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from data import mixup_data, cutmix_data, flip_data
from model import createResNet, ResNet34
from data import load_leaf_data
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resume = True
batchSize = 6
log = ["batchSize: %d\n" % batchSize]

testName = "eff4_331"

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
    # net = createResNet()
    # net = torchvision.models.resnet50(pretrained=False, num_classes=176)
    # net = EfficientNet.from_name('efficientnet-b4',  num_classes=176)
    net = EfficientNet.from_pretrained('efficientnet-b4',  num_classes=176)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=5e-4)
    net = net.to(device)
    if resume:
        print("加载模型。。。")
        net.load_state_dict(torch.load(r"E:\Data\20_epoch_eff4.pth"))
    # 定义两类损失函数
    loss = torch.nn.CrossEntropyLoss()
    criterion = mixup_criterion
    for epoch in range(num_epochs):
        startTime = time.time()
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        net.train()
        loss_total = 0
        for i, (features, labels) in enumerate(train_data):          
            trainer.zero_grad()
            X, Y = features.to(device), labels.to(device)
            # 数据增强
            random_num = np.random.random()
            if random_num <= 1/4:
                X, Y, y_b, lam = mixup_data(X, Y, use_cuda=True)
            elif random_num <= 2/4:
                X, Y, y_b, lam = cutmix_data(X, Y, use_cuda=True)
            elif random_num <= 3/4:
                X, Y, y_b, lam = flip_data(X, Y, use_cuda=True)
            else:
                X, Y, y_b, lam = mixup_data(X, Y, alpha=0, use_cuda=True)
            X, Y, y_b = map(torch.autograd.Variable, (X, Y, y_b))
            y_hat = net(X)            
            Y = Y.squeeze(dim=1)
            y_b = y_b.squeeze(dim=1)
            # l = loss(y_hat, Y)
            l = criterion(y_hat, Y, y_b, lam)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            loss_total = loss_total + l
        endTime = time.time()
        if epoch > 0 and epoch % 5 == 0:
            predictNet(net, test_data)
        # 每10轮保存一次结果
        if epoch > 0 and epoch % 10 == 0:
            torch.save(net.state_dict(), "E:\Data\%d_epoch_%s.pth" % (epoch, testName))
        log.append('epoch %d, loss %.4f, use time:%.2fs\n' % (
            epoch + 1, loss_total, endTime - startTime))        
        print('epoch %d, loss %.4f, use time:%.2fs' % (
            epoch + 1, loss_total, endTime - startTime))
    torch.save(net.state_dict(), "E:\Data\200_epoch_%s.pth" % testName)
    with open("E:\Data\train_log_%s.txt" % testName, "w+") as logFile:
        logFile.writelines(log)


def mixup_criterion(pred, y_a, y_b, lam):
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
    endTime = time.time()
    print("train acc: %.4f, use time:%.2fs" % (accNum / (len(test_data) * batchSize), endTime - startTime))
    log.append("train acc: %.4f, use time:%.2fs\n" % (accNum / (len(test_data) * batchSize), endTime - startTime))

if __name__ == "__main__":
    # net = createResNet()
    # print(net)
    # print(ResNet34())
    # print(torchvision.models.resnet50(pretrained=False, num_classes=176))
    train()