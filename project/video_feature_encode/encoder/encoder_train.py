#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:48:03 2018

@author: lps
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as dst
from torchvision.utils import save_image

from encoder_config import BATCH_SIZE, n, LATENT_CODE_NUM, log_interval, EPOCH, device
from encoder_model import VAE

transform=transforms.Compose([transforms.ToTensor()])
data_train = dst.MNIST(r'E:\Data\MLData\MNIST_data', train=True, transform=transform, download=True)
data_test = dst.MNIST(r'E:\Data\MLData\MNIST_data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=data_train, num_workers=n, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=data_test, num_workers=n, batch_size=BATCH_SIZE, shuffle=True)



def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x,  size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())      
    return BCE+KLD


def train():
    vae = VAE().to(device)
    optimizer =  optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for epoch in range(1, EPOCH):             
        vae.train()
        total_loss = 0
        for i, (data, _) in enumerate(train_loader, 0):
                data = Variable(data).to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar = vae.forward(data)
                loss = loss_func(recon_x, data, mu, logvar)
                loss.backward()
                total_loss += loss.data.item()
                optimizer.step()
                
                if i % log_interval == 0:
                    sample = Variable(torch.randn(64, LATENT_CODE_NUM)).to(device)
                    sample = vae.decoder(vae.fc2(sample).view(64, 128, 7, 7)).cpu()
                    save_image(sample.data.view(64, 1, 28, 28), r'E:\Data\MLData\MNIST_data\sample_' + str(epoch) + '.png')
                    print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                                epoch, i*len(data), len(train_loader.dataset), 
                                100.*i/len(train_loader), loss.data.item()/len(data)))
                    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(train_loader.dataset)))
      
      
if __name__ == "__main__":
    train()