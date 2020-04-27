# -!- encoding:utf-8 -!-

"""
主成分分析：
    1.标准化数据
    2.构建协方差矩阵
    3.将协方差矩阵分解为特征向量和特征值
    4.选择特征值最大的k个特征向量
    5.通过k个特征向量构建新的矩阵
    6.利用新创建的矩阵将原始的矩阵进行转换
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


# 读取txt文件的内容
def readtxt():
    f = open("F:/2017/Python/Data/Wine.txt")             # 返回一个文件对象
    line = f.readline()             # 调用文件的 readline()方法
    while line:
        print line,                 # 后面跟 ',' 将忽略换行符
        # print(line, end = '')　　　# 在 Python 3中使用
        line = f.readline()
    f.close()


# 获取数据可以从网上获取，也可以读取本地文件获取
def getdata():
    # df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    # print df_wine.head()
    df_wine1 = pd.read_csv('F:/2017/Python/Data/Wine.txt', header=None)
    print df_wine1.head()
    return df_wine1


if __name__ == '__main__':
    df_wine = getdata()
    # X为属性数据，y为label
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)
    # 1.标准化数据
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
    # 计算协方差，特征值越大其包含的信息量越大
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('\nEigenvalues \n%s' % eigen_vals)