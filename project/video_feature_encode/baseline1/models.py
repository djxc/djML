
import torch
import torch.nn as nn

# MLP模型思想为：
# 1、首先在每一帧进行提取参数，将其缩减一半宽度, 重复三次，宽度缩减为之前的8/1
# 2、然后在时间维度求均值，所有帧帧变为一个向量 
# 3、最后全连接获取结果。


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.d_input = args.d_input
        self.n_input = args.n_input
        self.output_size = args.num_classes
        print(f'MLPClassifier initialized, hidden configuration{[self.d_input, self.d_input//2, self.d_input//4, self.d_input//8, self.output_size, self.n_input]}')

        self.layer1 = torch.nn.Sequential(
            nn.Linear(self.d_input, self.d_input//2),       # linear可以输入二维数据，并不仅能输入一维数据，相应的weight应该也是二维的。
            nn.BatchNorm1d(self.n_input),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear(self.d_input//2, self.d_input//4),
            nn.BatchNorm1d(self.n_input),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        self.layer3 = torch.nn.Sequential(
            nn.Linear(self.d_input//4, self.d_input//8),
            nn.BatchNorm1d(self.n_input),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )

        self.classifier1 = nn.Linear(self.d_input//8, self.output_size)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.mean(dim=1)
        logit = self.classifier1(x).squeeze(-1)
        return logit

# 首先在时间轴方向进行提取数据，缩小数据量
class MLPlus(nn.Module):
    def __init__(self, args):
        super(MLPlus, self).__init__()
        self.d_input = args.d_input
        self.n_input = args.n_input
        self.output_size = args.num_classes
        print(f'MLPClassifier initialized, hidden configuration{[self.d_input, self.d_input//2, self.d_input//4, self.d_input//8, self.output_size, self.n_input]}')

        self.layer1 = torch.nn.Sequential(
            nn.Linear(self.n_input, self.n_input//2),       # linear可以输入二维数据，并不仅能输入一维数据，相应的weight应该也是二维的。
            nn.BatchNorm1d(self.d_input),
            nn.GELU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout()
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear(self.n_input//4, self.n_input//8),
            nn.BatchNorm1d(self.d_input),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.2)
        )
        self.frame_width = 16

        self.layer3 = torch.nn.Sequential(
            nn.Linear(self.d_input, self.d_input//2),
            nn.BatchNorm1d(self.frame_width),
            nn.GELU(),
            nn.Dropout(),
            nn.MaxPool1d(3, 2, 1),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Linear(self.d_input//4, self.d_input//8),
            nn.BatchNorm1d(self.frame_width),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.3),
        )

        self.layer5 = torch.nn.Sequential(
            nn.Linear(self.d_input//16, self.d_input//32),
            nn.BatchNorm1d(self.frame_width),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.2),
        )

        self.layer6 = torch.nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Dropout(p=0.2)
        )

        self.layer7 = torch.nn.Sequential(
            nn.Conv2d(4, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier1 = nn.Linear(self.d_input//64, self.output_size)


    def forward(self, x):
        x = torch.permute(x, (0,2,1))
        x = self.layer1(x)
        x = self.layer2(x)

        x = torch.permute(x, (0,2,1))
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)      # 16 * 32
        x = x.unsqueeze(dim=1)
        x = self.layer6(x)
        x = self.layer7(x)      # 8 * 1 * 4
        x = x.view(x.shape[0], -1)
        # x = x.mean(dim=1)
        logit = self.classifier1(x).squeeze(-1)
        return logit

# 时间维度增加全连接层，分别在时间维度以及帧维度增加注意力机制
# 1、首先在每一帧进行提取参数，将其缩减一半宽度, 重复三次，宽度缩减为之前的8/1 250*256
# 2、时间维度提取参数，将其衰减一半宽度，转置256*250，256*125，256*64
class MLPDJ(nn.Module):
    def __init__(self, args):
        super(MLPDJ, self).__init__()
        self.d_input = args.d_input
        self.n_input = args.n_input
        self.output_size = args.num_classes
        print(f'MLPClassifier initialized, hidden configuration{[self.d_input, self.d_input//2, self.d_input//4, self.d_input//8, self.output_size, self.n_input]}')

        self.layer1 = torch.nn.Sequential(
            nn.Linear(self.d_input, self.d_input//2),       # linear可以输入二维数据，并不仅能输入一维数据，相应的weight应该也是二维的。
            nn.BatchNorm1d(self.n_input),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.2),
        )
        self.layer2 = torch.nn.Sequential(
            nn.Linear(self.d_input//4, self.d_input//8),
            nn.BatchNorm1d(self.n_input),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.2),
        )
        self.layer3 = torch.nn.Sequential(
            nn.Linear(self.d_input//16, self.d_input//32),
            nn.BatchNorm1d(self.n_input),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.3),
        )

        self.frame_width = self.d_input//64

        self.layer4 = torch.nn.Sequential(
            nn.Linear(self.n_input, self.n_input//2),
            nn.BatchNorm1d(self.frame_width),
            nn.GELU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=0.2),
        )


        self.layer5 = torch.nn.Sequential(
            nn.Linear(self.n_input//4, self.n_input//8),
            nn.BatchNorm1d(self.frame_width),
            nn.GELU(),
            nn.MaxPool1d(3, 2, 1),
            nn.Dropout(p=0.3),
        )
        self.classifier1 = nn.Linear(16, self.output_size)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # batch维下的二维矩阵转置
        x = torch.permute(x, (0,2,1))
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.mean(dim=1) # batch_size * 256
        logit = self.classifier1(x).squeeze(-1)
        return logit

