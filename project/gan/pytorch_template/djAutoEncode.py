import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import cv2
from torchvision.transforms import transforms
import datetime
import numpy as np

learn_rate = 0.0001
batch_size = 20
step = [100, 400, 800, 1500]

def adjust_lr(epoch, optimizer):
    lr = learn_rate * (0.1 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    return lr

class DJAutoEncoder(nn.Module):
    def __init__(self):
        super(DJAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(128 * 128, 64 * 64),
            nn.Tanh(),
            nn.Linear(64 * 64, 32 * 32),
            nn.Tanh(),
            nn.Linear(32 * 32, 16 * 16),
            nn.Tanh(),
            nn.Linear(16 * 16, 8 * 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8 * 8, 16 * 16),
            nn.Tanh(),
            nn.Linear(16 * 16, 32 * 32),
            nn.Tanh(),
            nn.Linear(32 * 32, 64 * 64),
            nn.Tanh(),
            nn.Linear(64 * 64, 128 * 128),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        # print(encode.shape)
        decode = self.decoder(encode)
        # print(decode.shape)
        return encode, decode

def encode_conv(in_channel, out_channel, maxPoolSize):
    return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxPoolSize,stride=maxPoolSize),#16*11*11  
        )
def decode_conv(in_channel, out_channel, maxPoolSize):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=maxPoolSize, stride=maxPoolSize),#1*9*9
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

class DJConvAutoEncoder(nn.Module):
    '''卷积的自编码
        1、卷积操作比全连接参数更少，速度更快
    '''
    def __init__(self):
        super(DJConvAutoEncoder, self).__init__()
        self.encoder_conv1 = encode_conv(1, 16, 2)
        self.encoder_conv2 = encode_conv(16, 32, 2)
        self.encoder_conv3 = encode_conv(32, 64, 2)
        
        self.sig = nn.Sigmoid()
        
        self.decoder_conv1 = decode_conv(64, 32, 2)
        self.decoder_conv2 = decode_conv(32, 16, 2)
        self.decoder_conv3 = decode_conv(16, 1, 2)
    
    def forward111(self, x):      
        # print(x.shape)
        encode = self.encoder_conv1(x)
        print("encode1", encode.shape)
        encode = self.encoder_conv2(encode)
        print("encode1", encode.shape)
        encode = self.encoder_conv3(encode)
        encode = self.sig(encode)
        print("encode1", encode.shape)

        decode = self.decoder_conv1(encode)
        print("decode1", decode.shape)
        decode = self.decoder_conv2(decode)
        print("decode2", decode.shape)
        decode = self.decoder_conv3(decode)
        decode = self.sig(decode)
        print("decode3", decode.shape)
        return encode, decode
    
    def forward(self, x):      
        encode = self.encoder_conv1(x)
        encode = self.encoder_conv2(encode)
        encode = self.encoder_conv3(encode)
        # encode = self.sig(encode)

        decode = self.decoder_conv1(encode)
        decode = self.decoder_conv2(decode)
        decode = self.decoder_conv3(decode)
        decode = self.sig(decode)
        return encode, decode

class DJDataset(Dataset):
    '''数据加载类'''
    def __init__(self, root, dataSize, transform=None, target_transform=None, img_type=".jpg"):      
        self.imgPath = root
        self.transform = transform
        self.dataSize = dataSize
        self.img_type = img_type

    def __getitem__(self, index):
        '''根据index获取对应的图像'''
        imgPath = self.imgPath + str(index) + self.img_type
        img = cv2.imread(imgPath)
        img = img[:, :, 0]
        img = self.transform(img)      
        return img

    def __len__(self):
        '''返回数据的个数'''
        return self.dataSize

if __name__ == "__main__":
    ''''''
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomRotation(90)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.Normalize(0.5, 0.5),
    ])


    autoencode_dataset = DJDataset("/2020/data/car_test/autoencoder/train_", 100, transform=train_transforms, img_type=".png")
    dataloaders = DataLoader(autoencode_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据

    test_dataset = DJDataset("/2020/data/car_test/test_autoencoder/predict_", 5, transform=test_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)  # 使用pytorch的数据加载函数加载数据
    # autoencoder = DJAutoEncoder().cuda()
    autoencoder = DJConvAutoEncoder().cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    loss_func = nn.MSELoss().cuda()
    print("start train ...")
    for epoch in range(2000):
        # adjust_lr(epoch, optimizer)
        loss = 0
        i = 0
        starttime = datetime.datetime.now()
        totalLoss = 0
        for index, img in enumerate(dataloaders):
            img = img.cuda()
            # print(img.shape)
        # for i in range(100):
        #     imgPath = "/2020/data/car_test/train_" + str(i) + ".jpg"
        #     img = cv2.imread(imgPath)
        #     img = img[:, :, 0]
        #     img = img_transforms(img).cuda()
        #     # img = img.view(-1, 128 * 128).cuda()
        #     # print(img.shape, img.max(), img.min())
            encode, decode = autoencoder(img)
            # print(encode.shape, decode.shape)
            loss = loss_func(decode, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            # if i % 10 == 0:
            #     print("epoch: %d,\t img_name:%s,\t loss: %f,\t" %(epoch, imgPath, loss.item()))
        endtime = datetime.datetime.now()
        if epoch % 10 == 0:
            print("epoch: %d,\t spent time: %d,\t loss: %f" %(epoch, (endtime - starttime).seconds, totalLoss/100))
    autoencoder.eval()
    with torch.no_grad():
        for index, img in enumerate(test_dataloaders):
            img = img.cuda()
        # for i in range(5):
        #     imgPath = "/2020/data/car_test/predict_" + str(i) + ".jpg"
        #     img = cv2.imread(imgPath)
        #     img = img[:, :, 0]
        #     img = img_transforms(img).cuda()
        #     # img = img.view(-1, 128 * 128).cuda()
            encode, decode = autoencoder(img)            
            # decode = decode.view(128, 128).cpu().numpy()
            decode = torch.squeeze(decode)
            decode = decode.cpu().numpy()
            # newDecode = decode - 0.1
            print(decode.shape, decode.max(), decode.min(), decode.mean())
            mat_max = decode.max()
            mat_min = decode.min()
            decode = (250 * (decode - mat_min)/(mat_max - mat_min))
            cv2.imwrite("/2020/data/car_autoencoder/conv_" + str(index) + ".png", decode)