# GAN网络生成图片
# 1、分为两个部分分别训练模型
# 2、首先训练判别器，用真实数据利用判别器计算误差；在利用生成器生成假数据，
# 在利用判别器计算假数据的误差，俩误差相加然后更新判别器的参数；
# 3、训练生成器，利用随机数生成假数据，经过判别器判断数据，计算判别器判断
# 数据与真实数据之间的误差，更新生成器参数 #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt


batch_size = 128
num_epoch = 100
z_dimension = 100  # noise dimension

def tensor_to_np(tensor):
    '''tensor格式转换为numpy的图像格式，将第一个纬度去掉，然后跳转波段与行列顺序'''
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def MNISTData():
    # 1、ToTensor()把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray,
    # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    # 2、Normalize(),给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。
    # 即：Normalized_image=(image-mean)/std。
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5],
            [0.5])
        ])
    train_dataset = datasets.MNIST(root="/2020/data/", train=True, transform=data_tf)
    # test_dataset = datasets.MNIST(root="D:\\Data", train=False, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
 
 
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
 
 
class discriminator(nn.Module):
    '''判别器'''
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 
 
class generator(nn.Module):
    '''生成器'''
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )
 
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x
 

def train_gan():
    train_loader = MNISTData()
    D = discriminator().cuda()                  # discriminator model
    G = generator(z_dimension, 3136).cuda()     # generator model
    
    criterion = nn.BCELoss()                    # binary cross entropy
    
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
    
    # train
    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(train_loader):
            num_img = img.size(0)
            # =================train discriminator
            real_img = Variable(img).cuda()

            real_label = Variable(torch.ones(num_img)).cuda()
            fake_label = Variable(torch.zeros(num_img)).cuda()
    
            # compute loss of real_img
            real_out = D(real_img).squeeze()
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out  # closer to 1 means better
    
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, z_dimension)).cuda()
            fake_img = G(z)
            fake_out = D(fake_img).squeeze()
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better
    
            # bp and optimize
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
    
            # ===============train generator
            # compute loss of fake_img
            z = Variable(torch.randn(num_img, z_dimension)).cuda()
            fake_img = G(z)
            output = D(fake_img).squeeze()
            # print(real_label.shape, output.shape)
            g_loss = criterion(output, real_label)
    
            # bp and optimize
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
    
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                    'D real: {:.6f}, D fake: {:.6f}'
                    .format(epoch, num_epoch, d_loss.item(), g_loss.item(),
                            real_scores.data.mean(), fake_scores.data.mean()))
        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './dc_img/real_images.png')
    
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './dc_img/fake_images-{}.png'.format(epoch+1))
    
    torch.save(G.state_dict(), './generator.pth')
    torch.save(D.state_dict(), './discriminator.pth')

if __name__ == "__main__":
    train_gan()