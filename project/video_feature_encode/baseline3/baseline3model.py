import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
import torch#一个深度学习的库Pytorch
import torch.nn as nn#neural network,神经网络
import torch.nn.functional as F#神经网络函数库
import math

#设置随机种子
import random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device='cuda' if torch.cuda.is_available() else"cpu"


#位置编码:
class PositionalEncoding(nn.Module):
    #d_model是翻译的每个词或者每个字的嵌入维度,比如用[1,2,3,4]表示“我”,那它的d_model就是4
    def __init__(self,d_model=2048,dropout=0,max_len=250):
        #继承父类的属性和方法
        super(PositionalEncoding,self).__init__()
        #随机丢弃掉一部分的神经元
        self.dropout=nn.Dropout(p=dropout)
        #先初始化pe,max_len就是翻译的词向量的长度,d_model是维度,初始化为0,并移动到GPU上训练
        pe=torch.zeros(max_len,d_model).to(device)
        """
        torch.arange(0,5)->[0,1,2,3,4]
        然后再在指定位置上加一个维度[[0,1,2,3,4]]
        position:就是每个词的位置信息
        $PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$
        $PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$
        """
        position = torch.arange(0, max_len).unsqueeze(1)
        #10000^{-2i/d_model}=e^{-2i*ln(10000)/d_model}
        down=torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        #代入位置编码的公式,分别计算偶数位和奇数位的位置编码
        #pe[:,0::2]  :所有行从第0行开始步长为2的所有列的进行赋值
        pe[:, 0::2] = torch.sin(position * down)/10#为了保证位置编码不会占据信息的主要部分,故除以10
        pe[:, 1::2] = torch.cos(position * down)/10#为了保证位置编码不会占据信息的主要部分,故除以10
        #在位置编码外面再添加一个batch的维度,扩展成(1,max_len,d_model)的向量
        pe = pe.unsqueeze(0)
        """
        寄存器缓冲区,如果一个参数不用在梯度下降中被优化,又希望保存在模型中,可以用register_buffer.
        """
        self.register_buffer(name="pe", tensor=pe)
    def forward(self,x):
        """
        传入没有加上位置编码的词向量,传出加上位置信息的词向量
        requires_grad_(False)是不需要更新参数的意思
        pe[:, : x.size(1)]:取出pe的前x.size(1)列.
        广播成相同的形状.
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        """
        某些位置对应的编码值被随机地丢弃,减少过拟合.
        """
        return self.dropout(x) 
    
#多头注意力机制
class MultiHeadSelfAttention(nn.Module):
    #定义初始化函数,dim_in是
    def __init__(self,dim_in,d_model,num_heads=4):
        super(MultiHeadSelfAttention,self).__init__()
        self.dim_in=dim_in
        self.d_model=d_model
        self.num_heads=num_heads
        #向量的维度必须被头的个数整除,否则会抛出异常.
        assert d_model %num_heads==0,"d_model must be multiple of num_heads"
        #定义线性变换矩阵
        self.linear_q=nn.Linear(dim_in,d_model)
        self.linear_k=nn.Linear(dim_in,d_model)
        self.linear_v=nn.Linear(dim_in,d_model)
        self.scale=1/math.sqrt(d_model//num_heads)
        #最后的线性层
        self.fc=nn.Linear(d_model,d_model)
    def forward(self,x):
        batch,n,dim_in=x.shape
        assert dim_in==self.dim_in
        nh=self.num_heads
        dk=self.d_model//nh
        q=self.linear_q(x).reshape(batch,n,nh,dk).transpose(1,2)
        k=self.linear_k(x).reshape(batch,n,nh,dk).transpose(1,2)
        v=self.linear_v(x).reshape(batch,n,nh,dk).transpose(1,2)
        dist=torch.matmul(q,k.transpose(2,3))*self.scale
        dist=torch.softmax(dist,dim=-1)
        att=torch.matmul(dist,v)
        att=att.transpose(1,2).reshape(batch,n,self.d_model)
        output=self.fc(att)
        return output   
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #创建位置编码的类
        self.PositionalEncoding=PositionalEncoding()
        #创建自注意机制的类
        self.Self_Attention=MultiHeadSelfAttention(dim_in=36,d_model=36,num_heads=3)
        #通道1,用小卷积核提取局部特征.
        self.channel1=nn.Sequential(
            nn.Conv2d(500,128,1,1,0,bias=True),#32*32->32*32
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,3,1,0,bias=True),#32*32->30*30
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,0,bias=True),#30*30->28*28
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),#14*14  
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,3,1,0,bias=True),#12*12
            nn.Tanh(),
            nn.MaxPool2d(2,2,0),#6*6
            nn.BatchNorm2d(512),
        )
        #通道2,用大卷积核提取全局特征
        self.channel2=nn.Sequential(
            nn.Conv2d(500,128,7,1,0,bias=True),#32*32->26*26
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2,0),#13*13
            nn.Conv2d(128,256,5,1,0,bias=True),#13*13->9*9
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,4,1,0,bias=True),#6*6
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.Linear1=nn.Linear(1024*6*6,128)
        self.relu1=nn.ReLU()
        self.Linear2=nn.Linear(128,5)
    def forward(self,input): 
        output=input
        output=self.PositionalEncoding(output)#先给数据加上位置信息
        output=output.view(-1,500,32,32)#reshape
        #传入两个通道提取特征
        output=torch.cat((self.channel1(output),self.channel2(output)),dim=1)#1024*6*6
        output=output.view(-1,1024,6*6)#reshape
        output=output+self.Self_Attention(output)#跳连接
        output=output.view(-1,1024*6*6)#reshape
        output=self.Linear1(output)
        output=self.relu1(output)
        output=self.Linear2(output)
        return F.softmax(output,dim=1)
    
