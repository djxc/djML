import numpy as np#进行矩阵运算的库
import torch#一个深度学习的库Pytorch
import torch.nn as nn#neural network,神经网络
import torch.optim as optim#一个实现了各种优化算法的库
import torch.nn.functional as F#神经网络函数库
import re #导入正则表达式操作库
import os#与操作系统相关的库
#设置随机种子
import random

from baseline3model import CNN

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device='cuda' if torch.cuda.is_available() else"cpu"

def num_to_str(num):
    num=str(num)[::-1]
    for i in range(6-len(num)):
        num+="0"
    return num[::-1]

def get_npy(path):
    Files=os.listdir(path)
    return Files

def train():
    #获取训练集的标签
    train_list=""
    with open('train_list.txt','r') as f:
        train_list+=f.read()
    y_train=re.findall(r'"(\d)"',train_list)
    trainFiles=get_npy('train_feature')
    netC=CNN()
    train_accs=[]#存储训练集的准确率
    test_accs=[]#存储测试集的准确率
    #训练周期为40次
    num_epochs=40
    #优化器
    optimizer = optim.Adam(netC.parameters(),lr=0.0001,betas=(0.5,0.999))
    #损失函数
    criterion=nn.CrossEntropyLoss()
    netC=netC.to(device)
    for epoch in range(num_epochs):
        #训练
        netC.train()
        for left in range(0, 800, 50):
            train_X=[]
            train_y=[]
            random_num=[i for i in range(800)]
            np.random.shuffle(random_num)
            for index in range(left,left+50):
                #这时候是250*2048*1*1,需要先转成1*1*250*2048,然后转成250*2048
                array=np.load(f'data/视觉特征编码/train_feature/{num_to_str(random_num[index])}.npy')
                array=np.transpose(array,(2,3,0,1))[0][0]
                #数据增强,将一张图片变成4张图片
                train_X.append(array)
                train_y.append(int(y_train[random_num[index]]))
                train_X.append(np.flip(array, axis=1))
                train_y.append(int(y_train[random_num[index]]))
                train_X.append(np.flip(array, axis=0))
                train_y.append(int(y_train[random_num[index]]))
                array=np.flip(array, axis=0)
                train_X.append(np.flip(array, axis=1))
                train_y.append(int(y_train[random_num[index]]))
            train_X=np.array(train_X)#.reshape(-1,125,64,64)
            train_X=torch.Tensor(train_X).to(device)
            train_y=torch.Tensor(train_y).long().to(device)
            #将数据放进去训练
            output=netC(train_X).to(device)
            #计算每次的损失函数
            error=criterion(output,train_y).to(device)
            #反向传播
            error.backward()
            #优化器进行优化(梯度下降,降低误差)
            optimizer.step()
            #将梯度清空
            optimizer.zero_grad()
        print(f"epoch:{epoch},error:{error}")
        train_acc, verify_acc = verify(netC, y_train)
        train_accs.append(train_acc)
        test_accs.append(verify_acc)

def verify(netC, y_train):
    netC.eval()
    with torch.no_grad():
        pred_y=[]
        for index in range(len(y_train)):
            #这时候是250*2048*1*1,需要先转成1*1*250*2048,然后转成1*250*2048
            array=np.load(f'data/视觉特征编码/train_feature/{num_to_str(index)}.npy')
            array=np.transpose(array,(2,3,0,1))[0]#.reshape(-1,125,64,64)
            sample=torch.Tensor(array).to(device)
            output=netC(sample).to(device)
            output=output.detach().cpu().numpy()
            pred=np.argmax(output)
            pred_y.append(pred)
    train_acc=0
    for i in range(800):
        train_acc+=(pred_y[i]==int(y_train[i]))/800
    print("训练集上的准确率:",train_acc)
    verify_acc=0
    for i in range(800,len(y_train)):
        verify_acc += (pred_y[i]==int(y_train[i]))/100
    print("测试集上的准确率:",verify_acc)
    return train_acc, verify_acc

def test():
    pass

if __name__ == "__main__":
    train()
