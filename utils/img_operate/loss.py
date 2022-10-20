# -*- coding: utf-8 -*- 
'''
@FileDesciption 自定义实现损失函数
@Author small dj
@Date 2020-11-30
@LastEditor small dj
@LastEditTime 2020-11-30 21:37
'''
import torch

def cross_entropy_loss(predict, realValue):
    '''交叉熵损失函数
    1、将计算出来的各个类的概率与真实的类的one-hot(只有一个类值为1，其他类值为0)求交叉熵,其中p为预测的概率为softmax结果；
    q为真实的概率(经过one-hot编码)；由于当概率为0时会出现无穷大(小)情况因此需要将其修改torch.clip(predict, 0.0000001, 1-0.0000001)
    H(p, q) = -∑[px*log(qx) + (1-px)*log(1-qx)]
    由于以上方程不便于计算(标签很多为0结果为无穷数)修改为,当q为0也不至于出错
    H(p, q) = -∑[qx*log(px)]
    '''
    epsilon = 1e-10
    # realValue = torch.clip(realValue, epsilon, 1.0 - epsilon)
    realValue.requires_grad = True   

    # loss = torch.sum(predict * torch.log(realValue) + (1 - predict) * torch.log((1 - realValue)))
    loss = -torch.sum(realValue * torch.log(predict))

    return loss
