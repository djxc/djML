# -*- coding: utf-8 -*-
import numpy as np


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """遍历循环个数进行学习，每循环一次权重更新一次，计算cost"""
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def predict(self, X):
        """
        判断整合输入值是否大于0，对于零返回1，否则返回-1
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
