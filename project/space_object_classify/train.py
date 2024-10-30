import os
import time
from datetime import datetime
import torch
import torchvision
import numpy as np
import torch.nn as nn
# from data import mixup_data, cutmix_data, rotate_data
from model import ResNet
from data import load_space_object_data, model_root

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resume = True
batchSize = 4
learing_rate = 0.01
num_epochs = 300
model_name = "resnet"

log = ["{} batchSize: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batchSize)]
model_path = r"{}\{}".format(model_root, model_name)

if not os.path.exists(model_path):
    os.mkdir(model_path)

def train(augmentation=False):
    '''训练
        1、首先用restNet进行训练, batchSize=6学习率0.01，训练50轮，测试集精度0.88
            restNet训练, batchSize=12学习率0.01，训练200轮，测试集精度0.878
        2、数据增强，TODO
            增加mixup、cutmix以及flip三种增强方式，训练200轮，测试集精度0.94
        3、处理类别不平衡问题，欠采样-EasyEnsemble，从多类中有放回的采取少数类个数，与少数类形成多个数据集，训练多个模型，TODO
        4、增大batch_size会使学习速度下降，但可以提高模型的泛化性
        5、利用训练好的模型进行微调
    '''
    train_data, test_data = load_space_object_data(batchSize)
    best_acc = 0
    net = ResNet()
    
    trainer = torch.optim.SGD(net.parameters(), lr=learing_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=10, eta_min=0, last_epoch=-1)
    loss = torch.nn.CrossEntropyLoss()    

    log_file = open(r"{}\train_log_{}_pre.txt".format(model_root, model_name), "a+")
    data_size = len(train_data) * batchSize
    for epoch in range(num_epochs): 
        startTime = time.time()
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        net.train()
        loss_total = 0
        for i, (visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot) in enumerate(train_data):          
            trainer.zero_grad()
            visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot = visible_img_plus.to(device), sar_img_plus.to(device), cate_one_hot.to(device), zt_one_hot.to(device), zh_one_hot.to(device), fb_one_hot.to(device)           

            categeo_hat, zt_hat, zh_hat, fb_hat = net(visible_img_plus, sar_img_plus)            
                       
            cate_one_hot = cate_one_hot.squeeze(dim=1)
            zt_one_hot = zt_one_hot.squeeze(dim=1)
            zh_one_hot = zh_one_hot.squeeze(dim=1)
            fb_one_hot = fb_one_hot.squeeze(dim=1)

            categeo_loss = loss(categeo_hat, cate_one_hot)
            zt_loss = loss(zt_hat, zt_one_hot)
            zh_loss = loss(zh_hat, zh_one_hot)
            fb_loss = loss(fb_hat, fb_one_hot)
            l = categeo_loss + zt_loss + zh_loss + fb_loss
            l.backward()
            trainer.step()
            loss_total = loss_total + l

            if(i % 10 == 0):
                print(datetime.now(), "learning_rate:", scheduler.get_last_lr()[0], " ; loss:{}".format(loss_total / ((i + 1) * batchSize)))
        endTime = time.time()
        scheduler.step()
        if epoch > 0 and epoch % 5 == 0:
            acc = predictNet(net, test_data, log_file)
            if acc > best_acc:
                best_acc = acc
                save_net(net, "{}\\best_model.pth".format(model_path))
        # 每10轮保存一次结果
        if epoch > 0 and (epoch + 1) % 10 == 0:
            save_net(net, "{}\\{}_epoch.pth".format(model_path, epoch))
        log_info = 'epoch %d, loss %.4f, use time:%.2fs\n' % (epoch + 1, loss_total / data_size, endTime - startTime)
        log_file.write(log_info) 
        log.append(log_info)        
        print('epoch %d, loss %.4f, use time:%.2fs' % (
            epoch + 1, loss_total / data_size, endTime - startTime))
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


def data_augmentation(X, Y):
    # 数据增强
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    # X, Y, y_b, lam = cutmix_data(X, Y, use_cuda=use_cuda)
    X, Y, y_b, lam = mixup_data(X, Y, use_cuda=use_cuda)
    X, Y, y_b = map(torch.autograd.Variable, (X, Y, y_b))
    return X, Y, y_b, lam

def data_augmentation1(X, Y):
    # 数据增强
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    random_num = np.random.random()
    if random_num <= 1/4:
        X, Y, y_b, lam = mixup_data(X, Y, use_cuda=use_cuda)
    elif random_num <= 2/4:
        X, Y, y_b, lam = cutmix_data(X, Y, use_cuda=use_cuda)
    elif random_num <= 3/4:
        X, Y, y_b, lam = rotate_data(X, Y, use_cuda=use_cuda)
    else:
        X, Y, y_b, lam = mixup_data(X, Y, alpha=0, use_cuda=use_cuda)
    X, Y, y_b = map(torch.autograd.Variable, (X, Y, y_b))
    return X, Y, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    """损失函数"""
    c = nn.CrossEntropyLoss()    
    return lam * c(pred, y_a) + (1 - lam) * c(pred, y_b)


def predictNet(net, test_data, log_file):
    '''
        预测
    '''
    startTime = time.time()
    accNum = 0
    net.eval()
    categeo_acc, zt_acc, zh_acc, fb_acc = 0, 0, 0, 0

    for i, (visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot) in enumerate(test_data):     
        visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot = visible_img_plus.to(device), sar_img_plus.to(device), cate_one_hot.to(device), zt_one_hot.to(device), zh_one_hot.to(device), fb_one_hot.to(device)
        categeo_hat, zt_hat, zh_hat, fb_hat = net(visible_img_plus, sar_img_plus)   
        cate_one_hot = cate_one_hot.squeeze(dim=1)
        zt_one_hot = zt_one_hot.squeeze(dim=1)
        zh_one_hot = zh_one_hot.squeeze(dim=1)
        fb_one_hot = fb_one_hot.squeeze(dim=1)
        categeo_result = torch.eq(categeo_hat.max(1, keepdim=True)[1], cate_one_hot.max(1, keepdim=True)[1])
        zt_result = torch.eq(zt_hat.max(1, keepdim=True)[1], zt_one_hot.max(1, keepdim=True)[1])
        zh_result = torch.eq(zh_hat.max(1, keepdim=True)[1], zh_one_hot.max(1, keepdim=True)[1])
        fb_result = torch.eq(fb_hat.max(1, keepdim=True)[1], fb_one_hot.max(1, keepdim=True)[1])
        categeo_acc = categeo_acc + categeo_result.sum().item()
        zt_acc = zt_acc + zt_result.sum().item()
        zh_acc = zh_acc + zh_result.sum().item()
        fb_acc = fb_acc + fb_result.sum().item()
        accNum = accNum + (categeo_acc * 0.3 + zt_acc * 0.2 + zh_acc * 0.25 + fb_acc * 0.25) / 4
    use_time = time.time() - startTime
    acc = accNum / (len(test_data) * batchSize)
    log_str = "categeo_acc: %.4f, zt_acc: %.4f, zh_acc: %.4f, fb_acc: %.4f, train acc: %.4f, use time:%.2fs" % (categeo_acc, zt_acc, zh_acc, fb_acc, acc, use_time)
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
    train(augmentation=False)
    # verify()