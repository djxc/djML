import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader

from dataset import FaceDataset, dataset_collate
from model import TinySSD, multibox_target, multibox_prior
from util import process_bar, Accumulator, Timer, calculate_bbox, multibox_detection, bbox_to_rect, show_bboxes
from loss import calc_loss, cls_eval, bbox_eval

num_workers = 4
batch_size = 1


def load_face_data(data_root, batch_size):
    ''' 加载脸部数据集
    '''
    print("load train data, batch_size", batch_size)
    train_iter = dataloader.DataLoader(
        FaceDataset(True, data_root, train_fold=8), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers) # , collate_fn=dataset_collate)

    test_iter = dataloader.DataLoader(
        FaceDataset(False, data_root, train_fold=8), batch_size, drop_last=True,
        num_workers=num_workers) # , collate_fn=dataset_collate)
    return train_iter, test_iter


def show_face(images, faces_list):
    """显示图像，将人脸用框标识出来"""
    image_num = len(images)
    fig, ax = plt.subplots(1, image_num, figsize=(15, 5))
    for i in range(image_num):
        if image_num > 1:
            axi = ax[i]
        else:
            axi = ax
        image = images[i].numpy().transpose(1,2,0)
        axi.imshow(image)
        for bbox in faces_list[i]:
            patch = bbox_to_rect(bbox[1:], "r")
            axi.add_patch(patch)
        axi.set_title('image{}'.format(i))

def testAnchors(image_path):
    '''测试锚框
    '''
    image_size = 5
    sizes_list = [[0.04, 0.08, 0.1], [0.12, 0.16, 0.2], [0.3, 0.4, 0.5], [0.55, 0.65, 0.7],
         [0.75, 0.8, 0.85]]
    ratios_list = [[1, 1.5, 0.65]] * image_size
    img = Image.open(image_path)
    arry_img=np.asarray(img)
    h, w = arry_img.shape[:2]

    X = torch.rand(size=(1, 3, h, w))
    bbox_scale = torch.tensor((w, h, w, h))
    fig, ax = plt.subplots(1, image_size, figsize=(15, 5))
    for i in range(image_size):
        axi = ax[i]
        axi.imshow(img)
        axi.set_title('size{}'.format(sizes_list[i]))
        sizes = sizes_list[i]
        ratios = ratios_list[i]
        labels = []
        for i in range(len(sizes)):
            label = 's={}, r={}'.format(sizes[i], ratios[0])
            labels.append(label)
        for i in range(1, len(ratios)):
            label = 's={}, r={}'.format(sizes[0], ratios[i])
            labels.append(label)
        print(labels)
        # 根据大小以及长宽比生成锚框
        # Y = multibox_prior(X, sizes=sizes, ratios=ratios)
        Y = multibox_prior(X, sizes=sizes, ratios=ratios)
        boxes = Y.reshape(h, w, len(sizes) + len(ratios) - 1, 4)       # resize，5为每个像素锚框的个数，4为每个锚框的数据

        for j in range(1):
            i = (j + 1) * 200
            bbox = boxes[i, i, :, :] * bbox_scale   # size从大到小与ratio组合，然后是最小的size与ratio第一个元素之后的组合
            show_bboxes(axi, bbox, labels)
    plt.show()


def train(resume=False):
    """"""

    DATA_ROOT = r"D:\Data\MLData\facedata"
    train_iter, verify_iter = load_face_data(DATA_ROOT, batch_size)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.001, weight_decay=5e-4)
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
            # show_face(features, labels)
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
            process_bar((i + 1) / numData, cls_total / (i + 1), bbox_total / (i+1), None, epoch + 1, start_str='', end_str='100%', total_length=35)
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        if (epoch + 1) % 5 == 0:
            verify(net, verify_iter, device, epoch)
            # 每隔50次epoch保存模型
            torch.save(net.state_dict(), r'D:\Data\MLData\model\face_ssd_' + str(epoch + 1) + '.pkl')
            print("save model sucessfully")
        print("\nepoch:", epoch + 1, cls_err, bbox_mae)
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
        f'{str(device)}')
    
def verify(net, verify_iter, device, epoch):
    """"""
    net.eval()    
    numData = len(verify_iter)
    metric = Accumulator(4)
    with torch.no_grad(): 
        cls_total = 0
        bbox_total = 0 
        for i, (features, labels) in enumerate(verify_iter):
            X, Y = features.to(device), labels.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                    bbox_labels.numel())
            cls_total = cls_total + (1 - metric[0] / metric[1])
            bbox_total = bbox_total + metric[2] / metric[3]
            process_bar((i + 1) / numData, cls_total / (i + 1), bbox_total / (i+1), None, epoch + 1, start_str='', end_str='100%', total_length=35)
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]    
        print("\nverify:", epoch + 1, cls_err, bbox_mae)
    

def test(model_path):    
    DATA_ROOT = r"D:\Data\MLData\facedata"
    batch_size = 1
    train_iter, test_iter = load_face_data(DATA_ROOT, batch_size)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TinySSD(num_classes=1)
    net=net.to(device)
    net.load_state_dict(torch.load(model_path))
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
    net.eval()    
    threshold = 0.5
    with torch.no_grad(): 
        for i, (features, labels) in enumerate(test_iter):
            X, Y = features.to(device), labels.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, cls_preds, bbox_preds = net(X)
            cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
            output = multibox_detection(cls_probs, bbox_preds, anchors)
            idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
            output = output[0, idx]
            b, h, w = features[0].shape        
            bboxes = []
            for row in output:
                result = row.numpy()    # 第一个值为类别(0是背景，不显示)，第二个为得分，剩下的为bbox     
                cls = int(result[0])           
                score = float(row[1])
                if score < threshold:
                    continue           
                bbox = result[2:6] * (w, h, w, h)
                print("draw bbox", bbox, score, row[0])  
                bbox = np.insert(bbox, 0, cls)
                bboxes.append(bbox)
            show_face(X, [bboxes])      
    
def log_result(epoch, cls_err, bbox_mae):
    """"""
    print(f'epoch {epoch + 1}, class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')


    
if __name__ == "__main__":
    # train()
    # verify()
    test(r'D:\Data\MLData\model\face_ssd_55.pkl')
    # testAnchors(r"D:\Data\MLData\facedata\2003\08\01\big\img_14.jpg")