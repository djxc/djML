import torch
import torchvision.transforms as transforms

from data import CUB, splitUCMLanduseData, UCMLanduseDataset

def getCUBData():
    '''
    dataset = CUB(root='./CUB_200_2011')

    for data in dataset:
        print(data[0].size(),data[1])

    '''
    # 以pytorch中DataLoader的方式读取数据集
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    dataset = CUB(root='D:\\Data\\机器学习\\CUB_200_2011', is_train=True, transform=transform_train,)
    print(len(dataset))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)
    print(len(trainloader))

def getUCMLanduseData():
    ''''''
    trainDataset = UCMLanduseDataset("D:\\Data\\机器学习\\UCMerced_LandUse\\Images\\train_data.txt", None)
    trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=2, shuffle=True, num_workers=4,
                                              drop_last=True)
    epochNum = 10
    for epoch in range(epochNum):
        for label, image in trainloader:
            print(epoch, label, image.shape)

if __name__ == '__main__':
    # splitUCMLanduseData("D:\\Data\\机器学习\\UCMerced_LandUse\\Images")
    getUCMLanduseData()