import torch.nn as nn
import torch

class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction
 
    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x), -1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].reshape(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].reshape(len(x_hat), -1) - 0.5
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = mse / power
    return nmse