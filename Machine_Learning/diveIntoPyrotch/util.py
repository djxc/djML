from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import data
# 本函数已保存在d2lzh包中方方便便以后使用用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                data.sgd(params, lr, batch_size)
            else:
                optimizer.step() # “softmax回归的简洁实现”一节将用到
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' 
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


def evaluate_accuracy(data_iter, net):    
    '''模型评估
        1、遍历所有的数据，判读模型算出来的最大值与真实的label是否一致，如果一致则加一否则加0
    '''
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


def showIMG(bboxs=None):
    img=Image.open("D:\\ID.jpg")
    plt.figure(8)
    plt.imshow(img)
    if bboxs is not None:
        currentAxis=plt.gca()
        for bbox in bboxs:
            rect=patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r',facecolor='none')
            currentAxis.add_patch(rect)
    plt.show()