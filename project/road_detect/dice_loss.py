import torch
import torch.nn as nn
import torch.nn.functional as F
 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
 
    def forward(self, preds, targets):
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
 
        intersection = (preds_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
 
        return 1 - dice