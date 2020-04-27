# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """定义一个类，其中三个方法"""
    def __init__(self, n_iter, lr):
        """n_iter为循环运算次数， lr为学习率"""
        self.n_iter = n_iter
        self.learningRate = lr
        self.w0 = 0.01
        self.cost = []
        
    def fit(self, X, y_):        
        """1.根据X的维数构建权重初始值设为0.1
           2.在设定的循环周期内重复计算
           3.每个样本计算一次，更新一次权重值
        """
        self.w = np.ones(X.shape[1]) * 0.1
        for i in range(self.n_iter):    
            for x, target in zip(X, y_):   
                update = self.learningRate * (target - self.predict(x))
                self.w0 += update
                self.w += update * x
            print('权重为', self.w)

    def fitAdaline(self, X, y_):
        """adaline算法，使用连续值修改权重。
        全部样本运算结束后，更新权重
        """
        self.w = np.ones(X.shape[1]) * 0.01
        for i in range(self.n_iter):                               
            errors = y_ - self.calcul(X) 
            print('误差为', (errors**2).sum() / 2.0)
            self.w += self.learningRate * X.T.dot(errors)
            self.w0 += self.learningRate * errors.sum()     
            self.cost.append((errors**2).sum() / 2.0)
            print('权重为', self.w)
    
    def fitAdalineShuffle(self, X, y):
        """adaline算法，每次运算进行洗牌，可以加快拟合速度"""
        self.w = np.ones(X.shape[1]) * 0.1
        for i in range(self.n_iter):    
            X_, y_ = self.shuffle(X, y)
            cost_ = []
            for x, target in zip(X_, y_):   
                error = target - self.calcul(x) 
                self.w += self.learningRate * x.dot(error)
                self.w0 += self.learningRate * error    
                cost_.append(0.5 * (error**2))
            self.cost.append(sum(cost_)/len(y))
#                print('权重为', self.w)
            
    def predict(self, x):
        return np.where(self.calcul(x) > 0.0, 1, -1)

    def calcul(self, x):
         return np.dot(x, self.w) + self.w0
     
            
    def shuffle(self, X, y):
        """将数据顺序打乱"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
        
    def showCost(self):
        plt.plot(range(1, len(self.cost) + 1), self.cost, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average Cost')
        plt.show()
