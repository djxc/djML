# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class OperateData:
    """数据的基本操作，获取数据、显示数据、数据的标准化,正则化等等"""
    def __init__(self):
        self.age = 28
    
    def getData(self):
#        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        df = pd.read_csv('Data/test.csv')
        #print(df.tail())
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', 1, -1)
    #    print(y)
        x = df.iloc[0:100, [0, 2]].values
    #    print(x)
        return x, y
    
    
    def showData(self, X, y):
        plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue', marker='x', label='versicolor')
        plt.xlabel('petal length')
        plt.ylabel('sepal length')
        plt.legend(loc='upper left')
        plt.show()
        
    def standardization(self, X):
        """数据的z-score标准化：(X - mean)/ std"""
        X_std = np.copy(X)
        X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
        return X_std
    
    def standardizationSL(self, X_train, X_test):
        """使用scikit-learn库中的StandardScaler
            通过对train数据的fit将数据进行标准化
        """
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        return X_train_std, X_test_std
    
    def autoNorm(self, dataSet):
        """对数据离差标准化 (x-min)/(max-min)"""
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = np.zeros(np.shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - np.tile(minVals, (m,1))
        normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
        return normDataSet, ranges, minVals
