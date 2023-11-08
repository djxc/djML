# 定义损失函数
import torch
import torch.nn.functional as F


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
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def CrossEntropy(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

def calc_loss(cls_loss, bbox_loss, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls_preds = cls_preds.reshape(-1, num_classes)
    cls_labels = cls_labels.reshape(-1)
    cls = cls_loss(cls_preds, cls_labels).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    """由于类别预测结果放在最后一维， `argmax` 需要指定最后一维。"""
    return float(
        (cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    """bbox误差"""
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())