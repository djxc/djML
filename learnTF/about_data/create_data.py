# 生成机器学习需要的数据
# @author djxc
# @date 2020-02-03
import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split


class CreateData():
    def __init__(self):
        pass

    def himmelblau_data(self):
        '''生成himmelblau函数的数据，并进行预处理'''
        x = np.arange(-6, 6, 0.1)
        y = np.arange(-6, 6, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = self.__himmelblau([X, Y])
        return X, Y, Z

    def moons_data(self):
        '''通过sklearn生成形状类似月牙的非线性数据'''
        X, y = datasets.make_moons(n_samples=2000, noise=0.2, random_state=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X, y, X_train, X_test, y_train, y_test

    def __himmelblau(self, x):
        '''根据传入的list生成一个新的数据'''
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2
