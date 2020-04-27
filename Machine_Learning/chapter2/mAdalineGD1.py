# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import seed


class AdalineGD1(object):
    """
    eta: learning rate 
    n_iter: 循环次数
    shuffle: 是否进行打乱顺序，每循环一次就重新进行打乱顺序
    random_state: 随机种子在进行洗牌和初始化权重时使用到
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """按照循环次数进行学习，每循环一次计算所有样本的平均cost，
        在每次循环中，遍历整个样本，每计算一个样本就更新权重"""
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
               self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """打乱样本特征值"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """初始化权重值，这里都设为0"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """更新权重，如果计算的结果和实际结果不同，则更新权重值。计算cost值"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def predict(self, X):
        """
        执行预测分类，判断整合输入值是否大于0，对于零返回1，否则返回-1
        :param X: 
        :return: 
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def net_input(self, X):
        """
        将输入值进行整合，利用权重（去除第一个）和样本值点乘，加上权重值第一个
        在学习阶段直接返回结果作为预测结果，这是和perceptron最不同的地方
        :param X: 
        :return: 
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """在学习阶段用不到此函数，在进行测试的时候才用到"""
        return self.net_input(X)
