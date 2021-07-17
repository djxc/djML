# 定义损失函数
import torch

def squared_loss(y_hat, y):  # 损失函数
    '''平方损失函数
        1、真实label与计算的label相减，差的平方在移除2
        @param y_hat 计算出来的label
        @param y 真实的label
    '''
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def cross_entropy(y_hat, y):
    '''交叉熵损失函数
        1、首先将真实label转换为一行多列
        2、利用gather函数得出真实label对应计算出来的数值，然后求log
    '''
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))