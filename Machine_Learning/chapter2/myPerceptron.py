# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class myPerceptron(object):
    """
    eta为学习率， n_iter代表循环的次数
    w_表示权重值， errors代表每次学习后错误的大小
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_ter = n_iter

    def fit(self, X, y):
        """
        初始化权重，比样本维数大一个
        进行循环更新权重，计算错误率
        :param X: 样本值
        :param y: 真实值
        :return: 
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_ter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        """
        判断整合输入值是否大于0，对于零返回1，否则返回-1
        :param X: 
        :return: 
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        """
        将输入值进行整合，利用权重（去除第一个）和样本值点乘，加上权重值第一个
        :param X: 
        :return: 
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def plot_decision_regions(self, X, y, classifier, resolusion=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolusion), np.arange(x2_min, x2_max, resolusion))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(y)):
            print(idx, cl)
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
