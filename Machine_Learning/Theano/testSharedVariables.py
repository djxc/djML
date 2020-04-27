# -!- encoding:utf-8 -!-
from theano import shared
from theano import tensor as T
from theano import function
import numpy as np
import theano
import pandas as pd
import matplotlib.pyplot as plt
import perceptron

def testshared():
    # """
    #    1.shared为定义共享变量，不同函数之间可以相互访问此变量，修改其值为set_value(),获取其值get_value()
    #    2.updates函数需要共享变量与一个表达式，每次执行函数都会运行表达式修改共享变量
    # """
    state = shared(0)
    inc = T.iscalar('inc')
    accumulator = function([inc], state, updates=[(state, state + inc)])
    print(state.get_value())
    accumulator(2)
    print(state.get_value())


def Logistic_Regression():
    rng = np.random

    N = 40                                   # training sample size
    feats = 10                               # number of input variables

    # generate a dataset: D = (input_values, target_class)
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
    training_steps = 100000

    # Declare Theano symbolic variables
    x = T.dmatrix("x")
    y = T.dvector("y")

    # initialize the weight vector w randomly
    #
    # this and the following bias variable b
    # are shared so they keep their values
    # between training iterations (updates)
    w = theano.shared(rng.randn(feats), name="w")

    # initialize the bias term
    b = theano.shared(0., name="b")

    print("Initial model:")
    print(w.get_value())
    print(b.get_value())

    # Construct Theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
    prediction = p_1 > 0.5                    # The prediction thresholded
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
    cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
    gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                              # w.r.t weight vector w and
                                              # bias term b
                                              # (we shall return to this in a
                                              # following section of this tutorial)

    # Compile
    train = theano.function(
              inputs=[x, y],
              outputs=[prediction, xent],
              updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
    predict = theano.function(inputs=[x], outputs=prediction)

    # Train
    for i in range(training_steps):
        pred, err = train(D[0], D[1])

    print("Final model:")
    print(w.get_value())
    print(b.get_value())
    print("target values for D:")
    print(D[1])
    print("prediction on D:")
    print(predict(D[0]))


def mysigleneuron():
    x = T.vector()  #输入参数
    w = shared(np.array([1., 1.]))  #设为共享变量以便进行update修改,需要初始化赋值
    b = shared(0.)  #bais参数

    z = T.dot(x, w) + b  #计算矩阵相乘
    y = 1/(1 + T.exp(-z))  #使用sigmal函数计算结果
    y_hat = T.scalar()  #代表真实值

    cost = T.sum((y - y_hat)**2)
    getResult = theano.function([x], y)  #定义函数，
    dw, db = T.grad(cost, [w, b])  #计算cost对于w，b进行求导
    gradient = theano.function(inputs=[x, y_hat],   #通过update中的函数每运行一次就更新w，b
                               updates=MyUpdate([w, b], [dw, db]))
    x = [-1, 1]
    y_hat = 1
    for i in range(10000):
        y_pre = getResult(x)
        print(y_pre)
        if abs(y_pre - y_hat) < .01:
            print("ok")
            break

        gradient(x, y_hat)
        print(w.get_value(), b.get_value())

"""
1.首先计算出z=xw+b
2.根据计算出来的z通过激活函数计算预测的y
3.每循环一次就计算gradinent，该函数调用cost函数计算w，b的微分进行更新
"""
def mysigleneuron1(X, y1):
    x = T.vector()  #输入参数
    w = shared(np.array([1., 1.]))  #设为共享变量以便进行update修改,需要初始化赋值
    b = shared(0.)  #bais参数

    z = T.dot(x, w) + b  #计算矩阵相乘
    y = 1/(1 + T.exp(-z))  #使用sigmal函数计算结果
    y_hat = T.scalar()  #代表真实值

    cost = T.sum((y - y_hat)**2)
    getResult = theano.function([x], y)  #定义函数，
    dw, db = T.grad(cost, [w, b])  #计算cost对于w，b进行求导
    gradient = theano.function(inputs=[x, y_hat],   #通过update中的函数每运行一次就更新w，b
                               updates=MyUpdate([w, b], [dw, db]))
    for i in range(100):
        X, y1 = shuffle(X, y1)  #stochastic gradient随机从元数据集中抽取一个样本数据进行训练，因具有随机性可以使训练更有效，梯度下降的速度更快
        for xi, y_hat in zip(X, y1):
            y_pre = getResult(xi)
            print(y_pre - y_hat)
            gradient(xi, y_hat)
            # print(w.get_value(), b.get_value())

"""
随机返回数据集的数据，用来打乱顺序
"""
def shuffle(X, y):
    r = np.random.permutation(len(y))
    return X[r], y[r]

"""
标准化数据，每个特征的数据减去这个特征数据的平均然后除以标准差，在0附近成正态分布
标准化数据有利于计算
"""
def standdata(X):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std


def MyUpdate(paramters, gradients):
    mu = 0.1
    paramter_updates = [(p, p-mu*g) for p, g, in zip(paramters, gradients)]
    return paramter_updates


#从网上获取数据
def getData():
    num = 100
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # print(df.tail())  #读取最后五行
    print(df.head(55))  #读取前五行

    y = df.iloc[0:num, 4].values    #iloc函数是截取第四列，0->100行的数据返回一维数组:"Iris-setosa","Iris-versicolor"
    y = np.where(y == 'Iris-setosa', 0, 1) #where相当于判断语句，符合条件的设为-1，否则设为1

    X = df.iloc[0:num, [0, 2]].values

    return X, y


#将数据显示在面板上
def DrawData(X):
    plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')   #在画板上显示点，颜色为红色，样式为o，图例为setosa
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('petal length')  #设置横坐标的名称
    plt.ylabel('sepal length')  #设置纵坐标的名称
    plt.legend(loc='upper left')    #设置图例的位置
    plt.show()


if  __name__ == '__main__':
    # Logistic_Regression
    # mysigleneuron()
    X, y1 = getData()
    X_stand = standdata(X)
    # DrawData(X_stand)
    mysigleneuron1(X_stand, y1)
    # ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
    # ppn.fit(X, y1)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of misclassifications')
    # plt.show()
    # mysigleneuron1()
