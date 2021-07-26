import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import LeNet, AlexNet, VGG, createResNet, createDenseNet, TinySSD
from data import train_GPU, load_data_fashion_mnist, BananasDataset
from util import showIMG, multibox_prior, show_bboxes, multibox_target, multibox_detection, show_images
from util import Timer, Accumulator, Animator, set_figsize
from loss import calc_loss,cls_eval, bbox_eval

CURRENT_IMAGE_PATH = "/2020/"

def model_demo(net):
    batch_size = 156
    lr, num_epochs = 0.001, 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_GPU(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

def create_VGG():
    small_conv_arch = [
        (1, 1, 64),
        (1, 64, 128),
        (2, 128, 256),
        (2, 256, 512),
        (2, 512, 512)
    ]
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096
    net = VGG(small_conv_arch, fc_features, fc_hidden_units)
    return net


def display_anchors(img, w, h, fmap_w, fmap_h, s):
    fig = plt.imshow(img)
    # d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(fig.axes, anchors[0] * bbox_scale)
    plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg")

def multiLevelAnchors():
    '''多尺度锚框'''
    img = Image.open("/2020/213.jpg")
    arry_img=np.asarray(img)
    h, w = arry_img.shape[:2]
    display_anchors(img, w, h, fmap_w=1, fmap_h=1, s=[0.8])

def testAnchors():
    '''测试锚框
    '''
    img = Image.open("/2020/213.jpg")
    arry_img=np.asarray(img)
    h, w = arry_img.shape[:2]

    print(img.size, h, w)
    X = torch.rand(size=(1, 3, h, w))
    # 根据大小以及长宽比生成锚框
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = Y.reshape(h, w, 5, 4)
    print(boxes[250, 250,:,:])
       
    fig = plt.imshow(img)

    bbox_scale = torch.tensor((w, h, w, h))
    # show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, [
    #     's=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'
    # ])

    # 模拟真值以及锚框
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

    # show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    # show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

    labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
    print(labels)

    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    # show_bboxes(fig.axes, anchors * bbox_scale,
    #         ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                    offset_preds.unsqueeze(dim=0),
                    anchors.unsqueeze(dim=0), nms_threshold=0.5)
    for i in output[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
    plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg")


def load_data_bananas(batch_size):
    """加载香蕉检测数据集。"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

def bannasRecognition():
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)
    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, batch[1][0:10]):
        show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    num_epochs, timer = 10, Timer()
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                     legend=['class error', 'bbox mae'])
    net=net.to(device)
    
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    for epoch in range(num_epochs):
        # 训练精确度的和，训练精确度的和中的示例数
        # 绝对误差的和，绝对误差的和中的示例数
        metric = Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
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
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        # animator.add(epoch + 1, (cls_err, bbox_mae))
        print("epoch:", epoch + 1, cls_err, bbox_mae)
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
        f'{str(device)}')
    
    predictor = predictNet(net, device)
    X = torchvision.io.read_image('/2020/banna1.jpg').unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    output = predictor(X)
    display(img, output.cpu(), threshold=0.9)

def predictNet(net, device):
    def predict(X):
        net.eval()
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]
    return predict

def display(img, output, threshold):
    # set_figsize((5, 5))
    fig = plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg")


if __name__ == "__main__":
    # net = LeNet()
    # net = AlexNet()
    # net = create_VGG()
    # net = createResNet()
    # net = createDenseNet()
    # print(net)
    # model_demo(net)
    # showIMG(bboxs=[[250, 130, 350, 280], [150, 30, 250, 280]], save=True)
    # testAnchors()
    # multiLevelAnchors()
    bannasRecognition()
