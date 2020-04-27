import torch
import torchvision
import numpy as np
from myNet import Net
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


BATCH_SIZE = 4      # 定义每个batch的大小

def getCIFAR10Data():
    '''加载cifar10的数据， 返回训练数据、测试数据以及分类数据加载类，是可遍历的'''

    # 定义转换方法，将其转换为tensor，并进行标准化
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='D:/2020/python/data/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='D:/2020/python/data/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def imshow(img):
    '''将图像转换为矩阵，然后显示'''
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))       # 数组的转换，如果默认则为数据的转置
    plt.show()

def train(trainloader):
    '''训练模型，实例化模型、损失函数以及优化函数；最后保存模型'''
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # 循环两遍
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 输入的图像(矩阵)以及label
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    saveModel(net)
    print('Finished Training, save successfully!')

def saveModel(net):
    '''将训练好的模型保存为.pth文件，其中记录了模型的参数'''
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

def loadModel(PATH):
    '''根据输入的位置，加载模型'''
    net = Net()
    net.load_state_dict(torch.load(PATH))
    return net

def predict(net, images):
    '''对输入的一组图像进行预测'''
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

def testModel(net, testloader):
    '''测试模型，对所有的测试数据进行预测'''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

def showIMG(imgloader, classes):
    """随机获取训练图片,显示出来，imgloader属于迭代器
    每一批数据含有BATCH_SIZE各数据"""
    dataiter = iter(imgloader)
    images, labels = dataiter.next()
    print(images.size())
    # 打印图片标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))
    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    return images

if __name__ == "__main__":
    trainloader, testloader, classes = getCIFAR10Data()
    # train(trainloader)
    # net = loadModel('./cifar_net.pth')
    images = showIMG(testloader, classes)
    # predict(net, images)
    # testModel(net, testloader)