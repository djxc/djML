import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import TinySSD
from data import load_data_ITCVD
from util import show_bboxes, multibox_target, multibox_detection, Timer, Accumulator, process_bar, multibox_prior
from loss import calc_loss,cls_eval, bbox_eval, CrossEntropy

DATA_ROOT = "/2020/clothes_person/"
CURRENT_IMAGE_PATH = "/2020/"

def testAnchors():
    '''测试锚框
    '''
    sizes = [[0.01, 0.02], [0.037, 0.0447], [0.054, 0.0619], [0.071, 0.079],
         [0.088, 0.1]]
    ratios = [[1, 2, 0.5]] * 5
    img = Image.open("/2020/clothes_person/CARTclothes20200821_6.jpg")
    arry_img=np.asarray(img)
    h, w = arry_img.shape[:2]

    print(img.size, h, w, ratios)
    X = torch.rand(size=(1, 3, h, w))
    # 根据大小以及长宽比生成锚框
    Y = multibox_prior(X, sizes=[0.03, 0.04, 0.06, 0.08], ratios=[1, 2, 0.5])

    fig = plt.imshow(img)

    boxes = Y.reshape(h, w, 6, 4)       # resize，5为每个像素锚框的个数，4为每个锚框的数据
    print(boxes[250, 250,:,:])
       
    bbox_scale = torch.tensor((w, h, w, h))
    # for i in range(0, 4000, 800):
    i = 430
    show_bboxes(fig.axes, boxes[i, i, :, :] * bbox_scale, [
        # 's=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'
    ])
    plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg", dpi=580)

def trainITCVD(resume = False):
    '''训练ITCVD数据集'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, test_iter = load_data_ITCVD(DATA_ROOT, 1)
    net = TinySSD(num_classes=4)
    trainer = torch.optim.SGD(net.parameters(), lr=0.05, weight_decay=5e-4)
    num_epochs, timer = 200, Timer()   
    net=net.to(device)
    if resume:
        net.load_state_dict(torch.load('/2020/clothes_person_ssd.pkl'))
    # 定义两类损失函数
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss=nn.L1Loss(reduction='none')
    numData = len(train_iter)
    for epoch in range(num_epochs):
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        metric = Accumulator(4)
        net.train()
        cls_total = 0
        bbox_total = 0
        for i, (features, labels) in enumerate(train_iter):
            # print(features, labels)
            if labels.shape[1] == 0:
                continue
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), labels.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_loss, bbox_loss, cls_preds, cls_labels, bbox_preds, bbox_labels,
                        bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                    bbox_labels.numel())
            cls_total = cls_total + (1 - metric[0] / metric[1])
            bbox_total = bbox_total + metric[2] / metric[3]
            if i % 10 == 0:
                process_bar((i + 1) / numData, cls_total / (i + 1), bbox_total / (i+1), epoch + 1, start_str='', end_str='100%', total_length=35)
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        if (epoch + 1) % 10 == 0:
            # 每隔50次epoch保存模型
            torch.save(net.state_dict(), '/2020/clothes_person_ssd_' + str(epoch + 1) + '.pkl')
            print("save model sucessfully")
        # animator.add(epoch + 1, (cls_err, bbox_mae))
        print("\nepoch:", epoch + 1, cls_err, bbox_mae)
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
        f'{str(device)}')

def predictionITCVD():
    '''测试ssd'''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    device_cpu = torch.device('cpu')

    train_iter, test_iter = load_data_ITCVD(DATA_ROOT, 1)
    net = TinySSD(num_classes=4)
    net.load_state_dict(torch.load('/2020/clothes_person_ssd_last.pkl'))
    net=net.to(device)
    net.eval()
    # 定义两类损失函数
    for i, (features, labels) in enumerate(test_iter):
        if i > 10:
            break
        plt.cla()
        fig = plt.imshow(np.transpose(((features[0] + 2) * 50).int(), (1, 2, 0)))


        X, Y = features.to(device), labels.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)

        # output = multibox_detection(cls_probs.cpu(), bbox_preds.cpu(), anchors.cpu())
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        output = output[0, idx]
        threshold = 0.02
        b, w, h = features[0].shape
        print(i, b, w, h)
        plt.cla()
        fig = plt.imshow(np.transpose(features[0].int(), (1, 2, 0)))
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            bbox = [row[2:6] * torch.tensor((w, h, w, h), device=device_cpu)]        
            print("draw bbox", bbox, score)
            show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
            # show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
        plt.savefig(CURRENT_IMAGE_PATH + str(i) + "_Person_temp.jpg", dpi=580)

        # # 为每个锚框标注类别和偏移量
        # bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
        # print(anchors[0][0], cls_preds[0][0], bbox_preds[0][0])

if __name__ == "__main__":
    # trainITCVD()
    predictionITCVD()
    # testAnchors()