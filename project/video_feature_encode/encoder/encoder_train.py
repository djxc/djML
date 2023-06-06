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
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from encoder_config import BATCH_SIZE, n, LATENT_CODE_NUM, log_interval, EPOCH, device, workspace_root
from encoder_model import VAE
from encoder_data import get_mnist_data


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x,  size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())      
    return BCE+KLD


def train():
    data_train, data_test = get_mnist_data()
    train_loader = DataLoader(dataset=data_train, num_workers=n, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=data_test, num_workers=n, batch_size=BATCH_SIZE, shuffle=True)
    vae = VAE().to(device)
    optimizer =  optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for epoch in range(1, EPOCH):             
        vae.train()
        total_loss = 0
        for i, (data, _) in enumerate(train_loader, 0):
                data = Variable(data).to(device)    # variable 变量可以进行反向传播
                optimizer.zero_grad()
                recon_x, mu, logvar = vae.forward(data)
                loss = loss_func(recon_x, data, mu, logvar)
                loss.backward()
                total_loss += loss.data.item()
                optimizer.step()
                
                if i % log_interval == 0:
                    sample = Variable(torch.randn(64, LATENT_CODE_NUM)).to(device)  # torch.randn正态分布随机取数
                    sample = vae.decoder(vae.fc2(sample).view(64, 128, 7, 7)).cpu()
                    save_image(sample.data.view(64, 1, 28, 28), '{}\sample_{}.png'.format(workspace_root, epoch))
                    print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                                epoch, i*len(data), len(train_loader.dataset), 
                                100.*i/len(train_loader), loss.data.item()/len(data)))
                    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, total_loss / len(train_loader.dataset)))
      
      
if __name__ == "__main__":
    train()