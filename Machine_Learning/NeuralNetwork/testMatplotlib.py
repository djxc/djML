# -!- coding: utf-8 -！-
# %matplotlib inline
import matplotlib.pyplot as plt
import sys
from pandas import DataFrame  # DataFrame通常来装二维的表格
import pandas as pd  # pandas是流行的做数据分析的包
import testTheano as testT

plt.style.use('ggplot')
from sklearn import datasets
from sklearn import linear_model
import numpy as np

def drawpie():
    labels='frogs','hogs','dogs','logs'
    sizes=15,20,45,10
    colors='yellowgreen','gold','lightskyblue','lightcoral'
    explode=0,0.1,0,0
    plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
    plt.axis('equal')
    plt.show()


def drawpoint():
    # Load data
    boston = datasets.load_boston()
    yb = boston.target.reshape(-1, 1)
    Xb = boston['data'][:,5].reshape(-1, 1)
    # Plot data
    plt.scatter(Xb,yb)
    plt.ylabel('value of house /1000 ($)')
    plt.xlabel('number of rooms')
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit( Xb, yb)
    # Plot outputs
    plt.scatter(Xb, yb,  color='black')
    plt.plot(Xb, regr.predict(Xb), color='blue',
             linewidth=3)
    plt.show()


def drawdata():

    # print(df_miku.iloc[:, 0].values) # 读取第一列的值

    # print(df_miku.head())
    # print(df_miku.columns.size )     # 列数 2
    # print(point)
    # print(value)

    df_miku = pd.read_csv('F:/2017/Python/Data/Miku.txt', header=None, sep=' ')
    x, y, value = df_miku.iloc[:, 0].values, df_miku.iloc[:, 1].values, df_miku.iloc[:, 2].values
    # point, value = df_miku.iloc[:, :2].values, df_miku.iloc[:, 2].values

    pointx1 = []
    pointy1 = []
    pointx2 = []
    pointy2 = []

    for i in range(len(value)):
        if value[i] == 0:
            pointx1.append(x[i])
            pointy1.append(-y[i])
        else:
            pointx2.append(x[i])
            pointy2.append(-y[i])

    plt.scatter(pointx1, pointy1, color='black')
    plt.scatter(pointx2, pointy2, color='red')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()


def testdj():
    df_miku = pd.read_csv('F:/2017/Python/Data/Miku.txt', header=None, sep=' ')
    point, value = df_miku.iloc[:, :2].values, df_miku.iloc[:, 2].values
    # testT.dj(point, value, 250000, 1000)
    # testT.printdata(point, value, 250000, 1000)
    testT.drawMiku1(point, value)


def perprosessing():


    # 建立字典，键和值都从文件里读出来。键是nam，age……，值是lili，jim……
    dict_data = {}
    # 打开文件
    with open('file_in.txt', 'r')as df:
        # 读每一行
        for line in df:
            # 如果这行是换行符就跳过，这里用'\n'的长度来找空行
            if line.count('\n') == len(line):
                continue
                # 对每行清除前后空格（如果有的话），然后用"："分割
            for kv in [line.strip().split(':')]:
                # 按照键，把值写进去
                dict_data.setdefault(kv[0], []).append(kv[1])

                # print（dict_data）看看效果
    # 这是把键读出来成为一个列表
    columnsname = list(dict_data.keys())

    # 建立一个DataFrame，列名即为键名，也就是nam，age……
    frame = DataFrame(dict_data, columns=columnsname)

    # 把DataFrame输出到一个表，不要行名字和列名字
    frame.to_csv('file_out0.txt', index=False, header=False)

# drawdata()
testdj()