# -!- encoding:utf-8 -!-
import pandas as pd
import numpy as np
import math
from sklearn.cross_validation import train_test_split


# 获取数据可以从网上获取，也可以读取本地文件获取
def getdata():
    # df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    # print df_wine.head()
    df_wine1 = pd.read_csv('F:/2017/Python/Data/Wine.txt', header=None)
    print df_wine1.head(120)
    return df_wine1.head(120)

if __name__ == '__main__':
    df_wine = getdata()
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print X.shape[1]
    w = np.zeros(X.shape[1])
    eta = 0.01
    b = 0.1
    for _ in range(10):
        for x, y1 in zip(X, y):
            z = np.dot(x, w) + b
            a = 1/(1+math.exp(-z))

            a1 = np.where(a >= 0.5, 1, 2)
            err = y1 - a1
            update = eta*err
            w += update * x
            b += update
            # print err
