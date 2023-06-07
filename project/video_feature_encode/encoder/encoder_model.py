import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder_config import LATENT_CODE_NUM, device

class VAE(nn.Module):
      def __init__(self):
            super(VAE, self).__init__()
      
            self.encoder = nn.Sequential(
                  nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True),
                  
                  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True),
                      
                  nn.Conv2d(128, 128, kernel_size=3 ,stride=1, padding=1),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True),                  
                  )
            
            self.fc11 = nn.Linear(128 * 7 * 7, LATENT_CODE_NUM)
            self.fc12 = nn.Linear(128 * 7 * 7, LATENT_CODE_NUM)
            self.fc2 = nn.Linear(LATENT_CODE_NUM, 128 * 7 * 7)
            
            self.decoder = nn.Sequential(                
                  nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                  nn.ReLU(inplace=True),
                  
                  nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                  nn.Sigmoid()
                )

      def reparameterize(self, mu, logvar):
            """重参数
                  1、均值与方差有encoder计算处理，通过重参数对encoder进行反向优化
            """
            eps = torch.randn(mu.size(0), mu.size(1)).to(device)  # 随机生成数据，
            z = mu + eps * torch.exp(logvar/2)            
            
            return z
      
      def forward(self, x):
            out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 128, 7, 7
            mu = self.fc11(out1.view(out1.size(0),-1))     # batch_s, latent  均值
            logvar = self.fc12(out2.view(out2.size(0),-1)) # batch_s, latent  方差
            z = self.reparameterize(mu, logvar)      # batch_s, latent      
            out3 = self.fc2(z).view(z.size(0), 128, 7, 7)    # batch_s, 128, 7, 7
            
            return self.decoder(out3), mu, logvar
      
