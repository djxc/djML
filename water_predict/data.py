import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

LOOK_BACK = 7
IS_TRAIN = False
EPOCH_NUM = 1000


class lstm(nn.Module):
    '''定义lstm模型
        两个水厂，每个水厂进行一个lstm进行训练，将两者的和进行简单的相加与总用水量进行调整参数
        1、layer1为lstm层
        2、layer2为全连接层
        3、可以在进入lstm之前添加节假日的权重，暂未实现
    '''
    def __init__(self, input_size=LOOK_BACK, hidden_size=4, output_size=1, num_layer=2):
        super(lstm,self).__init__()
        # self.fistLayer = nn.Linear(7, )
        self.layer1 = nn.LSTM(3, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

        self.layer3 = nn.LSTM(3, hidden_size, num_layer)
        self.layer4 = nn.Linear(hidden_size, output_size)

        # self.layerOut = nn.Linear(719 * 2, 1)

    
    def forward(self, x_a, x_b):
        print(x_a.shape, x_b.shape)
        xa, _ = self.layer1(x_a)
        s, b, h = xa.size()
        xa = xa.view(s * b, h)
        xa = self.layer2(xa)
        xa = xa.view(s, b, -1)

        xb, _ = self.layer3(x_b)
        s, b, h = xb.size()
        xb = xb.view(s * b, h)
        xb = self.layer4(xb)
        xb = xb.view(s, b, -1)
        if IS_TRAIN:
            # print([xa, xb])
            # out = self.layerOut([xa, xb])
            # return out
            return xa,  xb
        else:
            return xa,  xb

def train_LSTM(train_a_x, train_a_y, train_b_x, train_b_y):
    '''创建lstm模型
        1、初始化模型
        2、定义损失函数以及优化函数
        3、循环进行训练
        @param train_x 训练数据
        @param train_y 样本
    '''
    model = lstm(LOOK_BACK, 4, 1, 2)
    criterion = nn.MSELoss()    # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)       # 优化函数
    # 开始训练
    for e in range(EPOCH_NUM):
        var_a_x = Variable(train_a_x)
        var_a_y = Variable(train_a_y)
        var_b_x = Variable(train_b_x)
        var_b_y = Variable(train_b_y)      
        # 前向传播
        outa, outb = model(var_a_x, var_b_x)
        loss = criterion(outa + outb, var_a_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (e + 1) % 10 == 0: # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
    torch.save(model, 'model.pkl')
    return model

def test_model(model, data_a_X, data_b_X):
    '''训练模型'''
    model = model.eval() # 转换成测试模式
    # data_a_X = data_a_X.reshape(-1, 1, LOOK_BACK)
    # data_b_X = data_b_X.reshape(-1, 1, LOOK_BACK)
    # data_X = torch.from_numpy(data_X)
    var_data_a = Variable(data_a_X)
    var_data_b = Variable(data_b_X)
    pred_text_a, pred_test_b = model(var_data_a, var_data_b) # 测试集的预测结果
    # 改变输出的格式
    pred_test_a = pred_text_a.view(-1).data.numpy()
    pred_test_b = pred_test_b.view(-1).data.numpy()
    for pred_a, pred_b in zip(pred_test_a, pred_test_b):
        print(pred_a, pred_b)

def max_min_standard(dataset):
    '''数据进行最大最小归一化'''
    # 数据进行归一化
    max_value = np.max(dataset)             # 获得最大值
    min_value = np.min(dataset)             # 获得最小值
    
    scalar = max_value - min_value          # 获得间隔数量
    dataset = list(map(lambda x: x / scalar, dataset)) # 归一化
    return dataset

def get_water_data(data_path):
    '''获取用水数据
        1、读取csv文件，获取每列数据，然后将a厂与b厂的供水数据与总供水量、节假日组合
    生成每条数据包含厂供水量、总供水量以及节假日数据
    '''
    water_data = pd.read_csv(data_path)
    dataset = water_data.values                     # 获得csv的值
    dataset = dataset.astype('float32')             # 将数据转换为浮点型
    dataset_a = dataset[:, 3]                       # 获取a厂数据
    dataset_b = dataset[:, 4]                       # 获取b厂数据
    dataset_total = dataset[:, 5]                   # 获取两厂总数据
    dataset_holiday = dataset[:, 7]                   # 获取节假日数据
    dataset_A = dataset_a.reshape(len(dataset_a), -1) # np.concatenate((dataset_a, dataset_holiday), axis=0)
    dataset_B = dataset_b.reshape(len(dataset_b), -1)
    dataset_Holiday = dataset_holiday.reshape(len(dataset_holiday), -1)
    dataset_Total = dataset_total.reshape(len(dataset_total), -1)
    dataSet_A = np.concatenate((dataset_A, dataset_Holiday, dataset_Total), axis=1)
    dataSet_B = np.concatenate((dataset_B, dataset_Holiday, dataset_Total), axis=1)
    # if standard:
    #     # 数据进行归一化,数据归一化是不能将测试数据进行归一化
    #     dataset_a = max_min_standard(dataset_a)
    #     dataset_b = max_min_standard(dataset_b)
    #     dataset_total = max_min_standard(dataset_total)
    return dataSet_A, dataSet_B


def get_fly_data():
    '''读取数据
        1、获取时间序列数据，将其转换为浮点型，进行最大最小值归一化处理
    '''
    fly_data = pd.read_csv("d://fly.csv")
    dataset = fly_data.values               # 获得csv的值
    dataset = dataset[:, 2]                 # 获取第三列数据
    max_min_standard(dataset)

def split_data(data_X, data_Y, train_precent = 0.7):
    '''划分训练集和测试集，70% 作为训练集
        1、拆分数据，
        2、只对训练集进行归一化,测试集不进行归一化
    '''
    train_size = int(len(data_X) * train_precent)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    # 归一化
    train_X = (train_X - train_X.mean(axis=0)) / train_X.std(axis=0)
    train_Y = (train_Y - train_Y.mean(axis=0)) / train_Y.std(axis=0)

    # 转换为模型可以识别的数据格式，转换为tensor类型
    # train_X = train_X.reshape(-1, 1, LOOK_BACK)
    # train_Y = train_Y.reshape(-1, 1, 1)
    # test_X = test_X.reshape(-1, 1, LOOK_BACK)

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_Y)
    return train_x, train_y, test_x, test_y

def create_dataset(dataset, look_back=LOOK_BACK):
    '''生成训练集
        1、遍历时间序列数据每look_back为一条数据， 第i+look_back的最后一个未为样本值
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back][-1])
    return np.array(dataX), np.array(dataY)


def train():
    ''''''
    print("读取数据")
    # fly_data = get_fly_data()
    dataset_A, dataset_B = get_water_data("d://water_train.csv")
    dataset_a_X, dataset_a_Y = create_dataset(dataset_A)
    dataset_b_X, dataset_b_Y = create_dataset(dataset_B)
    # print(dataset_b_X.shape, dataset_b_Y.shape)
    # dataset_total_X, dataset_total_Y = create_dataset(dataset_total)
    # dataset_holiday_X, dataset_holiday_Y = create_dataset(dataset_holiday)

    # # 创建好输入输出
    # # data_X, data_Y = create_dataset(fly_data)
    train_a_x, train_a_y, test_a_x, test_a_y = split_data(dataset_a_X, dataset_a_Y)
    train_b_x, train_b_y, test_b_x, test_b_y = split_data(dataset_b_X, dataset_b_Y)
    # train_total_x, train_total_y, test_total_x = split_data(dataset_total_X, dataset_total_Y)
    # train_holiday_x, train_holiday_y, test_holiday_x = split_data(dataset_holiday_X, dataset_holiday_Y)
    # print(train_holiday_x.shape)
    IS_TRAIN = True
    model = train_LSTM(train_a_x, train_a_y, train_b_x, train_b_y)
    return model, test_a_x, test_a_y, test_b_x, test_b_y

def test():
    ''''''
    IS_TRAIN = False

    test_dataset_a, test_dataset_b, test_dataset_total, test_dataset_holiday = get_water_data("d://water_test.csv", False)
    test_dataset_a_X, test_dataset_a_Y = create_dataset(test_dataset_a)
    test_dataset_b_X, test_dataset_b_Y = create_dataset(test_dataset_b)
    test_dataset_a_X, test_dataset_a_Y, test_data_a_test = split_data(test_dataset_a_X, test_dataset_a_Y, 1)
    test_dataset_b_X, test_dataset_b_Y, test_data_b_test = split_data(test_dataset_b_X, test_dataset_b_Y, 1)
    test_holiday_x, test_holiday_y, test_holiday_x = split_data(dataset_holiday_X, dataset_holiday_Y, 1)
    test_model(model, test_dataset_a_X, test_dataset_b_X, test_holiday_x)

if __name__ == "__main__":
    model, test_a_x, test_a_y, test_b_x, test_b_y = train()
    test_model(model, test_a_x, test_b_x)

    # print(train_x, train_y)
