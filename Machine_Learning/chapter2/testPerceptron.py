# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from myPerceptron import myPerceptron
from mAdalineGD import AdalineGD
from mAdalineGD1 import AdalineGD1


def show_origin_data():
    # 将第一个特征值为x，第二个为y，一个样本画出一个点。设置颜色样式和标签
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')      # 设置横纵坐标的标签
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')    # 设置图例的位置
    plt.show()


def test_perceptron():
    """
    进行训练样本，输出循环次数以及误差的关系
    """
    ppn = myPerceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    ppn.plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')    # 设置图例的位置
    plt.show()


def test_adaline(Xx, iter):
    """
    测试Adaline算法
    :param Xx: 样本特征值
    :param iter: 训练循环的次数
    :return: 
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    adal = AdalineGD(n_iter=iter, eta=0.01).fit(Xx, y)
    ax[0].plot(range(1, len(adal.cost_) + 1), np.log10(adal.cost_), marker='o')
    ax[0].set_xlabel('Epchos')
    ax[0].set_ylabel('Log(sum squared error)')
    ax[0].set_title('Learning rate 0.01')
    ada2 = AdalineGD(n_iter=iter, eta=0.0001).fit(Xx, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
    ax[1].set_xlabel('Epchos')
    ax[1].set_ylabel('Log(sum squared error)')
    ax[1].set_title('Learning rate 0.0001')
    plt.show()


if __name__ == "__main__":
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # df.to_csv("F:/test/test.csv", index=False, sep=',')
    df = pd.read_csv('F:/test/test.csv')
    print(df.tail())

    y = df.iloc[0:100, 4].values  # 截取行数：0-100， 列数：第四列。将值赋值给y，即为分类的目标类别
    print(y[48:52])
    y = np.where(y == 'Iris-setosa', -1, 1)  # 将y值进行量化，分为-1或是1，两个值
    print(y[48:52])
    X = df.iloc[0:100, [0, 2]].values  # 样本的特征值，这里取第一个和第三个
    print(X[0:5])
    # test_adaline(X, 20)
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()
    test_adaline(X_std, 20)

    ada = AdalineGD1(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average cost')
    plt.show()


