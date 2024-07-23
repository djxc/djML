
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from SSDModel import TinySSD
from data import CarDataset
from util import multibox_prior, show_bboxes, multibox_target, multibox_detection, show_images
from util import Timer, Accumulator
from loss import calc_loss,cls_eval, bbox_eval

CURRENT_IMAGE_PATH = r"D:\Data\MLData\车辆检测\car_det_train"

def load_data_car(batch_size):
    """加载香蕉检测数据集。"""
    train_iter = torch.utils.data.DataLoader(CarDataset(is_train=True, workspace=CURRENT_IMAGE_PATH),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(CarDataset(is_train=False, workspace=CURRENT_IMAGE_PATH),
                                           batch_size)
    return train_iter, val_iter

def display_anchors(img: Image.Image, fmap_w, fmap_h, s):
    """显示锚框
        @param img 图像
        @param w 图像宽度
    """
    fig = plt.imshow(img)
    # d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])

    bbox_scale = torch.tensor((img.width, img.height, img.width, img.height))
    show_bboxes(fig.axes, anchors[0] * bbox_scale)
    # plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg")
    plt.show()

def multiLevelAnchors(img_path):
    '''多尺度锚框'''
    img = Image.open(img_path)
    display_anchors(img, fmap_w=1, fmap_h=1, s=[0.5, 0.7])


def testAnchors(img_path):
    '''测试锚框
        锚框的个数与sizes和ratios有关，即为:len(sizes) + len(ratios) - 1
    '''
    sizes = [0.6, 0.4, 0.2]
    ratios = [1, 2, 0.5]
    img = Image.open(img_path)
    h, w = img.height, img.width

    print(img.size, h, w, ratios)
    X = torch.rand(size=(1, 3, h, w))
    # 根据大小以及长宽比生成锚框
    Y = multibox_prior(X, sizes=sizes, ratios=ratios)
    fig = plt.imshow(img)

    boxes = Y.reshape(h, w, len(sizes) + len(ratios) - 1, 4)       # resize，5为每个像素锚框的个数，4为每个锚框的数据
    print(boxes[250, 250,:,:])
       
    bbox_scale = torch.tensor((w, h, w, h))
    i = 120
    show_bboxes(fig.axes, boxes[i, i, :, :] * bbox_scale, ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
    # plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg", dpi=580)
    plt.show()


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
    # plt.savefig(CURRENT_IMAGE_PATH + "temp.jpg")
    plt.show()


def show_car_data():
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)

    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255   # 调整图像通道顺序，然后除以255，获取像素小数
    axes = show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, batch[1][0:10]):
        show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    plt.show()

def carRecognition():
    '''用ssd识别车辆数据集'''
    batch_size = 1
    train_iter, _ = load_data_car(batch_size) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    num_epochs, timer = 20, Timer()
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                     legend=['class error', 'bbox mae'])
    net=net.to(device)
    # 定义两类损失函数
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
        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), r'D:\Data\MLData\model\banna_ssd_' + str(epoch + 1) + '.pkl')
        # animator.add(epoch + 1, (cls_err, bbox_mae))
        print("epoch:", epoch + 1, cls_err, bbox_mae)
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
        f'{str(device)}')
    
    
   

def predict(net, X, device):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

def predictNet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    net = TinySSD(num_classes=1)
    net.load_state_dict(torch.load(r'D:\Data\MLData\model\banna_ssd_20.pkl'))
    for i in range(20):
        X = torchvision.io.read_image(r"D:\Data\MLData\MLData\detection\banana-detection\bananas_val\images\{}.png".format(i)).unsqueeze(0).float()
        img = X.squeeze(0).permute(1, 2, 0).long()
        output = predict(net, X, device)
        display(img, output.cpu(), threshold=0.9)


if __name__ == "__main__":
    img_path = os.path.join(CURRENT_IMAGE_PATH, "input_path", "1.tif")
    # multiLevelAnchors(img_path)
    # testAnchors(img_path)

    carRecognition()

    # predictNet()

    # show_banna_data()
