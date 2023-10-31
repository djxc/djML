import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.utils.data.dataloader as dataloader

from dataset import FaceDataset, dataset_collate
from model import TinySSD, multibox_target
from util import process_bar, Accumulator, Timer, calculate_bbox
from loss import calc_loss, cls_eval, bbox_eval


def load_face_data(data_root, batch_size):
    ''' 加载脸部数据集
    '''
    num_workers = 4
    print("load train data, batch_size", batch_size)
    train_iter = dataloader.DataLoader(
        FaceDataset(True, data_root), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, collate_fn=dataset_collate)

    test_iter = dataloader.DataLoader(
        FaceDataset(False, data_root, train_fold=2), batch_size, drop_last=True,
        num_workers=num_workers, collate_fn=dataset_collate)
    return train_iter, test_iter

def bbox_to_rect(bbox, color):
    # 将边界框 (左上x, 左上y, 右下x, 右下y) 格式转换成 matplotlib 格式：
    # ((左上x, 左上y), 宽, 高)
    return patches.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=1)


def show_face(images, faces_list):
    """显示图像，将人脸用框标识出来"""
    image_num = len(images)
    fig, ax = plt.subplots(1, image_num, figsize=(15, 5))
    for i in range(image_num):
        image = images[i].numpy().transpose(1,2,0)
        ax[i].imshow(image)
        for bbox in faces_list[i]:
            patch = bbox_to_rect(bbox[1:], "r")
            ax[i].add_patch(patch)
        ax[i].set_title('image{}'.format(i))


def train(resume=False):
    """"""

    DATA_ROOT = r"D:\Data\MLData\facedata"
    batch_size = 20
    train_iter, _ = load_face_data(DATA_ROOT, batch_size)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TinySSD(num_classes=2)
    trainer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.0005, weight_decay=5e-4)
    net=net.to(device)
    num_epochs = 200   
    if resume:
        net.load_state_dict(torch.load('/2020/clothes_person_ssd_last.pkl'))
    # 定义两类损失函数
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss=nn.L1Loss(reduction='none')
    numData = len(train_iter)
    timer = Timer()
    for epoch in range(num_epochs):
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        metric = Accumulator(4)
        net.train()
        cls_total = 0
        bbox_total = 0        
        for i, (features, labels) in enumerate(train_iter):
            timer.start()      
            # show_face(features, labels)
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
            # if i % 2 == 0:
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
    
if __name__ == "__main__":
    train()