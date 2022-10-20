import os
import torch
import time
import random
from torch import nn, optim
from collections import OrderedDict
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import torch.optim.lr_scheduler as lr_scheduler
from balanceGPU import BalancedDataParallel

from data import CUB, splitUCMLanduseData, UCMLanduseDataset, UsePhoneDataset, phone_dataset_collate
from model import LeNet, AlexNet, vgg11, createResNet, resnet50, resnet34, LabelSmoothSoftmaxCE
from util import process_bar

# 使用AlexNet时学习率设置的很大损失函数下降很慢，幅度很小；当减少学习率后损失函数下降很快
# 对于UCMLand数据集分类，不同模型的准确度
# 模型名称 训练次数 准确率 学习率
# lenet   50      0.19  0.0001
# alexnet 50      0.67  0.0001 
# alexnet 100     0.71  0.0001 
# vgg11   50      0.6   0.0001 
# resnet  50      0.8   0.0001 

# 使用第二块gpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getCUBData():
    '''
    dataset = CUB(root='./CUB_200_2011')

    for data in dataset:
        print(data[0].size(),data[1])

    '''
    # 以pytorch中DataLoader的方式读取数据集
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    dataset = CUB(root='D:\\Data\\机器学习\\CUB_200_2011', is_train=True, transform=transform_train,)
    print(len(dataset))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)
    print(len(trainloader))


def modelFactory(modelName, resume=False):
    '''模型工厂'''
    print("build mode: ", modelName)
    net = None
    if modelName == "LeNet":
        net = LeNet(3, 21).to(device)
    if modelName == "AlexNet":
        net = AlexNet(3, 21).to(device)
        if resume:
            net.load_state_dict(torch.load('/2020/' + modelName + '_ucmlanduse_50.pkl')) 
    elif modelName == "VGG":
        net = vgg11(num_classes=21).to(device)
    elif modelName == "ResNet":
        net = createResNet(3, 21).to(device)
    return net


def trainUCMLanduseData(modelName="LeNet", resume = False):
    '''训练UCMLanduse土地利用分类数据'''
    trainDataset = UCMLanduseDataset("/2020/data/landuse/UCMerced_LandUse/train_data.txt", None)
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=4,
                                              drop_last=True)
    net = modelFactory(modelName, resume) 
    if net is None:
        print("can not build model, exit")
        return
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    epochNum = 100
    imageNum = len(trainloader)
    for epoch in range(epochNum):
        loss_sum = 0
        for i, (labels, images) in enumerate(trainloader):
            net.train()
            images, labels = images.to(device), labels.to(device)
            prect_labels = net(images)
            # print(prect_labels)
            print(labels)
            loss = criteon(prect_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            loss_sum = loss_sum + loss.item()
            optimizer.step()
            if (i + 1) % 5 == 0:
                process_bar((i + 1) / imageNum, loss_sum / (i + 1), 0, epoch)
        print(epoch, 'loss:', loss_sum/len(trainloader))
        if (epoch + 1) % 10 == 0:
            valUCMLanduseData(net)
    torch.save(net.state_dict(), '/2020/' + modelName + '_ucmlanduse_50.pkl')
    valUCMLanduseData(net)


def valUCMLanduseData(net=None, modelName="LeNet"):
    if net is None:
        print("load model...")
        net = AlexNet(3, 21).to(device)
        net.load_state_dict(torch.load('/2020/' + modelName + '_ucmlanduse_50.pkl'))
        net = net.to(device)
    valDataset = UCMLanduseDataset("/2020/data/landuse/UCMerced_LandUse/val_data.txt", None)
    valloader = torch.utils.data.DataLoader(valDataset, batch_size=4, shuffle=True, num_workers=4,
                                              drop_last=True)
    acc_sum = 0
    net.eval()
    for label, image in valloader:
        image, label = image.to(device), label.to(device)
        predict_label = net(image)
        acc_sum += (predict_label.argmax(dim=1) == label).float().sum().cpu().item()
    print('acc:', acc_sum / len(valDataset))


def trainUsePhone(resume=False, isGPUs=False):
    ''''''
    modelName = "resnet34"
    trainDataset = UsePhoneDataset("/2020/data/usePhone/train/trainFiles.txt", isTrain=True)
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=1, shuffle=True, num_workers=4,
                                              drop_last=True) # , collate_fn=phone_dataset_collate)

    # 多卡运行，
    net = resnet34(num_classes=2) # .to(device)
    # net = nn.DataParallel(net).cuda()
    if resume:
        print("load model %s..." %modelName)
        # 先加载模型参数dict文件
        state_dict = torch.load('/2020/usePhoneModelReNet/' + modelName + '_usePhone_last.pkl')
        if isGPUs:
            net = BalancedDataParallel(0, net, dim=0).cuda()       # 解决多卡数据分配不均匀问题，第一个参数为第一个卡数据的batch_size
            from collections import OrderedDict
            # 初始化一个空 dict
            new_state_dict = OrderedDict()
            # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v
            # 加载修改后的新参数dict文件
            net.load_state_dict(new_state_dict)
        else:
            net = net.cuda()
            net.load_state_dict(state_dict)
    if net is None:
        print("can not build model, exit")
        return
    criteon = nn.CrossEntropyLoss().cuda() #.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-9)
    epochNum = 400
    imageNum = len(trainloader)
    # --------------------------------------
    # valUsePhoneData(net)
    # return
    # --------------------------------------

    for epoch in range(185, epochNum):
        loss_sum = 0
        for i, (labels, images) in enumerate(trainloader):
            # print(labels, images.dtype)
            net.train()
            # images, labels = images.to(device), labels.to(device)
            images, labels = images.cuda(), labels.cuda()
            prect_labels = net(images)           
            loss = criteon(prect_labels, labels)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            loss_sum = loss_sum + loss.item()
            optimizer.step()
            images = None
            if (i + 1) % 5 == 0:
                process_bar((i + 1) / imageNum, loss_sum / (i + 1), 0, epoch, total_length=50)
        print("\nepcho:", epoch + 1,'; loss:', loss_sum/len(trainloader), "; time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))        
        valUsePhoneData(net)
        torch.save(net.state_dict(), '/2020/usePhoneModelReNet/' + modelName + '_usePhone_' + str(epoch + 1) + '.pkl')


def valUsePhoneData(net, modelName="LeNet"):
    if net is None:
        print("no net, exit")
        return 
    
    valDataset = UsePhoneDataset("/2020/data/usePhone/train/valFiles.txt", False)
    valloader = torch.utils.data.DataLoader(valDataset, batch_size=1, shuffle=True, num_workers=4,
                                              drop_last=True, collate_fn=phone_dataset_collate)
    acc_sum = 0
    TP = 0    # 正样本正确总数
    TN = 0    # 负样本正确总数
    FP = 0    # 正样本错误总数
    FN = 0    # 负样本错误总数
    torch.cuda.empty_cache()
    net.eval()
    for label, image in valloader:
        # image, label = image.to(device), label.to(device)
        image, label = image.cuda(), label.cuda()
        predict_label = net(image)
        # 计算准确率以及得分，得分通过
        current_acc = (predict_label.argmax(dim=1) == label).float().sum().cpu().item()
        # 如果current_acc大于0则表示分类正确,否则为错误
        if current_acc > 0:
            if label.item() > 0:
                TN += 1
            else:
                TP += 1
        else:
            if label.item() > 0:
                FN += 1
            else:
                FP += 1
        acc_sum += current_acc
    # 计算精准率与召回率：精准率又叫查准率Precision在所有被预测为正的样本中实际为正的样本的概率
    # 召回率（Recall）又叫查全率，实际为正的样本中被预测为正样本的概率
    phonePrecision = TP/(TP + FP)
    phoneRcall = TP / (TP + FN)
    noPhonePrecision = TN/(TN + FN)
    noPhoneRcall = TN / (TN + FP)
    phoneScore = 2 * phonePrecision * phoneRcall / (phonePrecision + phoneRcall)
    noPhoneScore = 2 * noPhonePrecision * noPhoneRcall / (noPhonePrecision + noPhoneRcall)
    score = 0.6 * phoneScore + 0.4 * noPhoneScore
    print('acc:', acc_sum / len(valDataset), " ;TP: ", TP, " ;TN:", TN," ;FP: ", FP, " ;FN:", FN, " ;score", score)

def predUsePhone():
    # images = os.listdir("/2020/data/usePhone/test_images_a/")
    # with open("/2020/data/usePhone/predictImage.txt", "w") as predIMGs:
    #     for img in images:
    #         predIMGs.write(img + "\n")

    modelName = "resnet34"
    trainDataset = UsePhoneDataset("/2020/data/usePhone/predictImage.txt", False, True)
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=1, num_workers=4)

    net = resnet34(num_classes=2).cuda() #.to(device)
    net.load_state_dict(torch.load('/2020/usePhoneModelReNet/' + modelName + '_usePhone_last.pkl')) 
    net.eval()
    imageNum = len(trainloader)

    with open("/2020/data/usePhone/predictLabel.txt", "w") as predLabels:
        for i, (label, image) in enumerate(trainloader):
            image = image.cuda() #.to(device)
            imageName = label[0]
            predict_label = net(image)
            predict_label = predict_label.argmax(dim=1).cpu().item()
            predLabels.write(imageName + "," + str(predict_label) + "\n")
            if (i + 1) % 5 == 0:
                process_bar((i + 1) / imageNum, 0, 0, 0, total_length=50)
            


def efficientnet_train():
    modelName = "efficientnet-b4"
    # 需要分类的数目
    num_classes = 2
    # 批处理尺寸
    batch_size = 1  
    # 训练多少个epoch
    EPOCH = 150
    # feature_extract = True ture为特征提取，false为微调
    feature_extract = False
    # 超参数设置
    pre_epoch = 0  # 定义已经遍历数据集的次数


    input_size = 224  
    # EfficientNet的使用和微调方法
    net = EfficientNet.from_pretrained('efficientnet-b4')
    net._fc.out_features = num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load('/2020/efficientnet-b4_usePhone__net_best.pth')) 
    net = net.to(device)
    # 数据预处理部分
    train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])

    val_transforms = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainDataset = UsePhoneDataset("/2020/data/usePhone/train/trainFiles.txt", None, transforms=train_transforms)
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                              drop_last=True) #, collate_fn=phone_dataset_collate)
    valDataset = UsePhoneDataset("/2020/data/usePhone/train/valFiles.txt", None, transforms=val_transforms)
    valloader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                              drop_last=True) #, collate_fn=phone_dataset_collate)
  
    params_to_update = net.parameters()

    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    else:
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                # print("\t", name)
                pass

    ii = 0
    LR = 1e-3  # 学习率
    best_acc = 0  # 初始化best test accuracy
    print("Start Training, DeepNetwork!")  # 定义遍历数据集的次数
       
    # criterion
    criterion = LabelSmoothSoftmaxCE()
    # optimizer
    optimizer = optim.Adam(params_to_update, lr=LR, betas=(0.9, 0.999), eps=1e-9)
    # scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                # scheduler.step(epoch)
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                imageNum = len(trainloader)
                for  i, (labels, images) in enumerate(trainloader):
                    # 准备数据
                    images, labels = images.to(device), labels.to(device)
                    # 训练
                    optimizer.zero_grad()
                    # forward + backward
                    output = net(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    if (i + 1) % 10 == 0:
                        print('[epoch:%d, iter:%d, process:%.03f] Loss: %.03f | Acc: %.3f%% '
                            % (epoch + 1, (i + 1 + epoch * imageNum), 100. * (i+1) / imageNum, sum_loss / (i + 1),
                                100. * float(correct) / float(total)))
                        f2.write('%03d  %05d %.03f|Loss: %.03f | Acc: %.3f%% '
                                % (epoch + 1, (i + 1 + epoch * imageNum), 100. * (i+1) / imageNum, sum_loss / (i + 1),
                                    100. * float(correct) / float(total)))
                        f2.write('\n')
                        f2.flush()
                    

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for label, image in valloader:
                        net.eval()
                        images, labels = image.to(device), label.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).cpu().sum()
                    print('accuracy: %.3f%%' % (100. * float(correct) / float(total)))
                    acc = 100. * float(correct) / float(total)
                    scheduler.step(acc)

                    # 将每次测试结果实时写入acc.txt文件中
                    if (ii % 1 == 0):
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s_net_%03d.pth' % ('/2020/' + modelName + '_usePhone_',  epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


def val_effecientNet():
    input_size = 224  
    val_transforms = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valDataset = UsePhoneDataset("/2020/data/usePhone/predictImage.txt", None, True, transforms=val_transforms)
    valloader = torch.utils.data.DataLoader(valDataset, batch_size=1, shuffle=True, num_workers=4,
                                              drop_last=True) #, collate_fn=phone_dataset_collate)
    # EfficientNet的使用和微调方法
    net = EfficientNet.from_pretrained('efficientnet-b4')
    net._fc.out_features = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load('/2020/efficientnet-b4_usePhone__net_best.pth')) 
    net = net.to(device)
    net.eval()
    imageNum = len(valloader)

    with open("/2020/data/usePhone/predictLabel.txt", "w") as predLabels:
        for i, (label, image) in enumerate(valloader):
            image = image.to(device)
            imageName = label[0]
            predict_label = net(image)
            predict_label = predict_label.argmax(dim=1).cpu().item()
            predLabels.write(imageName + "," + str(predict_label) + "\n")
            if (i + 1) % 5 == 0:
                process_bar((i + 1) / imageNum, 0, 0, 0, total_length=50)

if __name__ == '__main__':
    # splitUCMLanduseData("D:\\Data\\机器学习\\UCMerced_LandUse\\Images")
    # trainUCMLanduseData("ResNet")
    # valUCMLanduseData()
    # net = vgg11(num_classes=21)
    # print(net)
    trainUsePhone(True, True)
    # predUsePhone()
    # efficientnet_train()
    # val_effecientNet()
    # phoneImage = os.listdir("/2020/data/usePhone/train/0_phone/JPEGImages/")
    # with open("/2020/data/usePhone/train/phoneFile.txt", "r") as phoneFile:
    #     images = phoneFile.readlines()
    #     random.shuffle(images)
    #     print(images[10].replace("\n", ""))


