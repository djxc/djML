import torch
from torch import mean, log, rand
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np


def encode_conv(in_channel, out_channel, maxPoolSize):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxPoolSize,stride=maxPoolSize),#16*11*11  
        )
def decode_conv(in_channel, out_channel, maxPoolSize):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=maxPoolSize, stride=maxPoolSize),#1*9*9
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )



class DJConvAutoEncoder(nn.Module):
    '''卷积的自编码
        1、卷积操作比全连接参数更少，速度更快
    '''
    def __init__(self):
        super(DJConvAutoEncoder, self).__init__()
        self.encoder_conv1 = encode_conv(1, 16, 2)
        self.encoder_conv2 = encode_conv(16, 32, 2)
        self.encoder_conv3 = encode_conv(32, 64, 2)
        
        self.sig = nn.Sigmoid()
        
        self.decoder_conv1 = decode_conv(64, 32, 2)
        self.decoder_conv2 = decode_conv(32, 16, 2)
        self.decoder_conv3 = decode_conv(16, 1, 2)
    
    def forward111(self, x):      
        # print(x.shape)
        encode = self.encoder_conv1(x)
        print("encode1", encode.shape)
        encode = self.encoder_conv2(encode)
        print("encode1", encode.shape)
        encode = self.encoder_conv3(encode)
        encode = self.sig(encode)
        print("encode1", encode.shape)

        decode = self.decoder_conv1(encode)
        print("decode1", decode.shape)
        decode = self.decoder_conv2(decode)
        print("decode2", decode.shape)
        decode = self.decoder_conv3(decode)
        decode = self.sig(decode)
        print("decode3", decode.shape)
        return encode, decode
    
    def forward(self, x):      
        encode = self.encoder_conv1(x)
        encode = self.encoder_conv2(encode)
        encode = self.encoder_conv3(encode)
        # encode = self.sig(encode)

        decode = self.decoder_conv1(encode)
        decode = self.decoder_conv2(decode)
        decode = self.decoder_conv3(decode)
        decode = self.sig(decode)
        return encode, decode

class EncodeNet(nn.Module):
    def __init__(self):
        super(EncodeNet, self).__init__()
        self.encoder_conv1 = encode_conv(1, 2, 2)
        self.encoder_conv2 = encode_conv(2, 4, 2)
        self.encoder_conv3 = encode_conv(32, 64, 2)

    def forward(self, x):
        encode = self.encoder_conv1(x)
        # print("encode 1 -----------", encode.shape)
        encode = self.encoder_conv2(encode)
        # print("encode 2 -----------", encode.shape)
        # encode = self.encoder_conv3(encode)
        return encode

class DecodeNet(nn.Module):
    def __init__(self):
        super(DecodeNet, self).__init__()
        self.decoder_conv1 = decode_conv(64, 32, 2)
        self.decoder_conv2 = decode_conv(4, 2, 2)
        self.decoder_conv3 = decode_conv(2, 1, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # decode = self.decoder_conv1(x)
        decode = self.decoder_conv2(x)
        # print("decode 1 -----------", decode.shape)

        decode = self.decoder_conv3(decode)
        # print("decode 2 -----------", decode.shape)
        decode = self.sig(decode)
        return decode

# q(z|x)
class Q_net(nn.Module):  
    '''编码器'''
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3_gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        z_gauss = self.lin3_gauss(x)
        return z_gauss

# p(x|z)
class P_net(nn.Module):  
    '''解码器'''
    def __init__(self,X_dim,N,z_dim):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

# D()
class D_net_gauss(nn.Module):  
    '''判别器'''
    def __init__(self, N, z_dim):
        super(D_net_gauss, self).__init__()
        # self.lin1 = nn.Linear(z_dim, N)
        # self.lin2 = nn.Linear(N, N)
        # self.lin3 = nn.Linear(N, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 16, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 784),
            nn.LeakyReLU(0.2, True),
            nn.Linear(784, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        # x = F.relu(x)
        # x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        # x = F.relu(x)
        # return F.sigmoid(self.lin3(x))  
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x         


def getData():
    # MNIST Dataset 
    dataset = dsets.MNIST(root="/2020/data/", 
                        train=True, 
                        transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=100, 
                                            shuffle=True)
    return data_loader

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return V(x)  

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

def train():
    data_loader = getData()
    EPS = 1e-15
    # 学习率
    gen_lr = 0.0001
    reg_lr = 0.00005
    # 隐变量的维度
    z_red_dims = 120
    # encoder
    # Q = Q_net(784, 1000, z_red_dims).cuda()
    Q = EncodeNet().cuda()
    # decoder
    # P = P_net(784, 1000, z_red_dims).cuda()
    P = DecodeNet().cuda()
    # discriminator
    D_gauss = D_net_gauss(500,z_red_dims).cuda()


    #encode/decode 优化器
    optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
    optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
    # GAN部分优化器
    optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
    optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)

    # 数据迭代器
    data_iter = iter(data_loader)
    iter_per_epoch = len(data_loader)
    total_step = 50000

    for step in range(total_step):
        if (step+1) % iter_per_epoch == 0:
            data_iter = iter(data_loader)
        # 从MNSIT数据集中拿样本
        images, labels = next(data_iter)
        images = images.cuda()
        # images, labels = to_var(images.view(images.size(0), -1)), to_var(labels)
        # 把这三个模型的累积梯度清空
        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()
        ################ Autoencoder部分 ######################
        # 1、编码器根据真实数据生成编码之后数据，然后利用解码器解码，与真实数据计算loss，
        # 分更新编码器与解码器

        # encoder 编码x, 生成z
        z_sample = Q(images)
        # decoder 解码z, 生成x'
        X_sample = P(z_sample)
        # print(images.shape, z_sample.shape, X_sample.shape)
        # 这里计算下autoencoder 的重建误差|x' - x|
        recon_loss = F.binary_cross_entropy(X_sample + EPS, images + EPS)

        # 优化autoencoder
        recon_loss.backward()
        optim_P.step()
        optim_Q_enc.step()

        ################ GAN 部分 #############################
        # 1、首先从真实数据送入判别器，计算真实数据判别结果，然后将编码后的数据放入判别器计算判别结果，
        # 利用两次判别误差优化判别器
        # 2、利用编码器对真实数据进行编码，放入判别器，判别器计算判别结果，然后优化编码器

        # 从正太分布中, 采样real gauss(真-高斯分布样本点)
        z_real_gauss = V(torch.tensor(np.random.randn(images.size()[0], 4, 7, 7) * 5.)).cuda()
        # 判别器判别一下真的样本, 得到loss
        # print(z_real_gauss.float())
        D_real_gauss = D_gauss(z_real_gauss.float())

        # 用encoder 生成假样本
        Q.eval()  # 切到测试形态, 这时候, Q(即encoder)不参与优化
        z_fake_gauss = Q(images)
        # print(z_fake_gauss.shape)
        # 用判别器判别假样本, 得到loss
        D_fake_gauss = D_gauss(z_fake_gauss)
        # 判别器总误差
        D_loss = -mean(log(D_real_gauss + EPS) + log(1 - D_fake_gauss + EPS))

        # 优化判别器
        D_loss.backward()
        optim_D.step()

        # encoder充当生成器
        Q.train()  # 切换训练形态, Q(即encoder)参与优化
        z_fake_gauss = Q(images)
        D_fake_gauss = D_gauss(z_fake_gauss)

        G_loss = -mean(log(D_fake_gauss + EPS))

        G_loss.backward()
        # 仅优化Q
        optim_Q_gen.step()
        if step == 0:
            real_images = to_img(images.cpu().data)
            save_image(real_images, './aae_cnn1/real_images.png')
        if (step+1) % 100 == 0:
            print(G_loss.item(), "---------")
            fake_images = to_img(X_sample.cpu().data)
            save_image(fake_images, './aae_cnn1/fake_images-{}.png'.format(step+1))
            # print(z_fake_gauss.shape, D_fake_gauss.shape, X_sample.shape)

    # 训练结束后, 存一下encoder的参数
    torch.save(Q.state_dict(), 'Q_encoder_weights.pt')

if __name__ == "__main__":
    print("start train ...")
    train()