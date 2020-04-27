import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
import AdaptiveLinearNeuron

# 从网上获取数据
def getData():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print(df.tail())  # 读取最后五行
    # print(df.head(55))  # 读取前55行
    y_ = df.iloc[0:100, 4].values    # iloc函数是截取第四列，0->100行的数据返回一维数组:"Iris-setosa","Iris-versicolor"
    y_ = np.where(y_ == 'Iris-setosa', -1, 1)  # where相当于判断语句，符合条件的设为-1，否则设为1
    # print(y)

    X_ = df.iloc[0:100, [0, 2]].values

    return X_, y_


# 将数据显示在面板上
def DrawData(X):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')   # 在画板上显示点，颜色为红色，样式为o，图例为setosa
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

    plt.xlabel('petal length')  # 设置横坐标的名称
    plt.ylabel('sepal length')  # 设置纵坐标的名称
    plt.legend(loc='upper left')    # 设置图例的位置
    plt.show()


# 进行分类
def RunClassifies():
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

# 画出分类的界限
def DrawLine():
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    ppn.plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


def testAdaLine():
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdaptiveLinearNeuron.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    ada2 = AdaptiveLinearNeuron.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()


def Standardi():
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()  # [:,0],逗号左边数字表示行，逗号右侧数字表示列
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std


def Standardization():
    # #先对数据进行归一化处理然后进行分类
    X_std=Standardi()
    ada = AdaptiveLinearNeuron.AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    perceptron.Perceptron.plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()


def testAdalineSGD():
    X_std = Standardi()
    ada = AdaptiveLinearNeuron.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    perceptron.Perceptron.plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochastic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()

if  __name__ == '__main__':
    X, y = getData()   # X为基础数据，y为标签
    # DrawData(X)
    RunClassifies()
    DrawLine()
    # testAdaLine()
    # Standardization()
    # testAdalineSGD()