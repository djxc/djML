import os
import time
from datetime import datetime
import torch
import torchvision
import numpy as np
import torch.nn as nn
# from data import mixup_data, cutmix_data, rotate_data
from model import ResNet
from data import load_space_object_data, load_test_space_object_data, rootPath, model_root, object_infos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resume = True
batchSize = 4
learing_rate = 0.001
num_epochs = 100
model_name = "resnet-se"

log = ["{} batchSize: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batchSize)]
model_path = r"{}\{}".format(model_root, model_name)

if not os.path.exists(model_path):
    os.mkdir(model_path)

def train(fold):
    '''训练
        1、首先用restNet进行训练, batchSize=6学习率0.01，训练50轮，测试集精度0.88
            restNet训练, batchSize=12学习率0.01，训练200轮，测试集精度0.878
        2、数据增强，TODO
            增加mixup、cutmix以及flip三种增强方式，训练200轮，测试集精度0.94
        3、处理类别不平衡问题，欠采样-EasyEnsemble，从多类中有放回的采取少数类个数，与少数类形成多个数据集，训练多个模型，TODO
        4、增大batch_size会使学习速度下降，但可以提高模型的泛化性
        5、利用训练好的模型进行微调
    '''
    print("start {} fold train".format(fold + 1))
    train_data, test_data = load_space_object_data(batchSize, fold)
    best_acc = 0
    net = ResNet().to(device)
    load_net(net)
    trainer = torch.optim.SGD(net.parameters(), lr=learing_rate, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=20, eta_min=0, last_epoch=-1)
    loss = torch.nn.CrossEntropyLoss()    

    log_file = open(r"{}\train_log_{}_pre_fold{}.txt".format(model_root, model_name, fold), "a+")
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
            l = categeo_loss * 0.3 + zt_loss * 0.2 + zh_loss * 0.25 + fb_loss * 0.25
            l.backward()
            trainer.step()
            loss_total = loss_total + l

            if(i % 30 == 0):
                # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "learning_rate:%.4f; loss:%.6f" % (scheduler.get_last_lr()[0], loss_total / ((i + 1) * batchSize)))
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " loss:%.6f" % (loss_total / ((i + 1) * batchSize)))

        endTime = time.time()
        # scheduler.step()       
        acc = predictNet(net, test_data, log_file)
        if acc > best_acc:
            best_acc = acc
            save_net(net, "{}\\best_model_fold{}.pth".format(model_path, fold))
        # 每10轮保存一次结果
        if epoch > 0 and (epoch + 1) % 10 == 0:
            save_net(net, "{}\\{}_epoch_fold{}.pth".format(model_path, epoch, fold))
        log_info = 'epoch %d, loss %.4f, use time:%.2fs\n' % (epoch + 1, loss_total / data_size, endTime - startTime)
        log_file.write(log_info) 
        log.append(log_info)        
        print('epoch %d, loss %.4f, use time:%.2fs' % (
            epoch + 1, loss_total / data_size, endTime - startTime))
    log_file.close()

def load_net(net):
    """加载模型的参数"""
    best_model_path = "{}\\best_model2.pth".format(model_path)
    if os.path.exists(best_model_path):
        print("加载模型。。。")
        if device.type == "cpu":
            net.load_state_dict(torch.load(best_model_path, map_location ='cpu'))
        else:
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
    net.eval()
    categeo_acc, zt_acc, zh_acc, fb_acc = 0, 0, 0, 0
    test_data_size = (len(test_data) * batchSize)

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
    use_time = time.time() - startTime

    acc = categeo_acc/test_data_size * 0.3 + zt_acc/test_data_size * 0.2 + zh_acc/test_data_size * 0.25 + fb_acc/test_data_size * 0.25
    log_str = "categeo_acc: %.4f, zt_acc: %.4f, zh_acc: %.4f, fb_acc: %.4f, train acc: %.4f, use time:%.2fs" % (categeo_acc/test_data_size, 
                                                                                                                zt_acc/test_data_size, 
                                                                                                                zh_acc/test_data_size, 
                                                                                                                fb_acc/test_data_size, 
                                                                                                                acc, use_time)
    print(log_str)
    log_file.write(log_str)
    return acc

def verify():
    """"""
    train_data, test_data = load_space_object_data(1)
    net = ResNet().to(device)
    load_net(net)
    net.eval()
    categeo_acc, zt_acc, zh_acc, fb_acc = 0, 0, 0, 0
    accNum = 0
    for i, (visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot) in enumerate(test_data):     
        visible_img_plus, sar_img_plus, cate_one_hot, zt_one_hot, zh_one_hot, fb_one_hot = visible_img_plus.to(device), sar_img_plus.to(device), cate_one_hot.to(device), zt_one_hot.to(device), zh_one_hot.to(device), fb_one_hot.to(device)
        categeo_hat, zt_hat, zh_hat, fb_hat = net(visible_img_plus, sar_img_plus)   
        cate_one_hot = cate_one_hot.squeeze(dim=1)
        zt_one_hot = zt_one_hot.squeeze(dim=1)
        zh_one_hot = zh_one_hot.squeeze(dim=1)
        fb_one_hot = fb_one_hot.squeeze(dim=1)
        zt_result = torch.eq(zt_hat.max(1, keepdim=True)[1], zt_one_hot.max(1, keepdim=True)[1])
        zh_result = torch.eq(zh_hat.max(1, keepdim=True)[1], zh_one_hot.max(1, keepdim=True)[1])
        fb_result = torch.eq(fb_hat.max(1, keepdim=True)[1], fb_one_hot.max(1, keepdim=True)[1])
        zt_acc = zt_acc + zt_result.sum().item()
        zh_acc = zh_acc + zh_result.sum().item()
        fb_acc = fb_acc + fb_result.sum().item()

        # 由于主体个数、载荷以及帆板较为准确因此需要进行对比
        zt_zh_fb_pred = "{}{}{}".format(zt_hat.max(1, keepdim=True)[1].item(), zh_hat.max(1, keepdim=True)[1].item(), fb_hat.max(1, keepdim=True)[1].item())
        max_value, max_index = categeo_hat.max(1, keepdim=True)
        print(max_value)
        zt_zh_fb_cate = object_infos[max_index.item()]
        if zt_zh_fb_pred != zt_zh_fb_cate:
            tensor = categeo_hat.masked_fill(categeo_hat == max_value.view(-1, 1), float('-inf'))
            tmp_max_value, tmp_max_index = tensor.max(1, keepdim=True)
            if (tmp_max_value - max_value).item() < 1:
                max_index = tmp_max_index

        categeo_result = torch.eq(max_index, cate_one_hot.max(1, keepdim=True)[1])
        categeo_acc = categeo_acc + categeo_result.sum().item()
        if not categeo_result.item():
            print("error: ", max_index.item() + 1, cate_one_hot.max(1, keepdim=True)[1].item() + 1)
        accNum = accNum + (categeo_acc * 0.3 + zt_acc * 0.2 + zh_acc * 0.25 + fb_acc * 0.25) / 4
    acc = accNum / (len(test_data) * batchSize)
    log_str = "categeo_acc: %.4f, zt_acc: %.4f, zh_acc: %.4f, fb_acc: %.4f, train acc: %.4f" % (categeo_acc, zt_acc, zh_acc, fb_acc, acc)
    print(log_str)  
    return 


def test():
    """"""
    test_data = load_test_space_object_data(1)
    net = ResNet().to(device)
    load_net(net)
    net.eval()
    result = []
    for i, (visible_img_plus, sar_img_plus, folder_name) in enumerate(test_data):     
        visible_img_plus, sar_img_plus = visible_img_plus.to(device), sar_img_plus.to(device)
        categeo_hat, zt_hat, zh_hat, fb_hat = net(visible_img_plus, sar_img_plus)   

        categeo_value, categeo_index = categeo_hat.max(1, keepdim=True)
        zt_result = zt_hat.max(1, keepdim=True)[1]
        zh_result = zh_hat.max(1, keepdim=True)[1]
        fb_result = fb_hat.max(1, keepdim=True)[1]
        categeo_index = int(categeo_index.item()) + 1
        print(categeo_index, zt_result.item(), zh_result.item(), fb_result.item(), categeo_hat)
        if categeo_value < 5:
            categeo_index = 0
        result.append("{}\t{}\t{}\t{}\t{}\n".format(folder_name[0], categeo_index, zt_result.item(), zh_result.item(), fb_result.item()))

    result_path = os.path.join(rootPath, "result-{}.csv".format(datetime.now().strftime("%Y%m%dT%H%M%S")))
    with open(result_path, "w") as result_file:
        result_file.writelines(result)
    return 

if __name__ == "__main__":
    # net = createResNet()
    # print(net)
    # print(ResNet34())
    # print(torchvision.models.resnet50(pretrained=False, num_classes=176))
    for i in range(5):
        train(i)
    # test()
    # verify()