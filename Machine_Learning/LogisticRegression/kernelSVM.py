#-！- coding:utf-8 -!-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import myData


def Drawdata():
    plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],c='r', marker='s', label='-1')
    plt.ylim(-3.0)
    plt.legend()
    plt.show()

def Drawclass():
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    myData.plot_decision_regions(X_xor, y_xor, classifier=svm,test_idx=range(105,150))
    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)  # 随机生成200个两列的数组
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)  # 判断数组第一、二列是否都大于0，
    y_xor = np.where(y_xor, 1, -1)  # 如果都大于0，设为-1，否则设为1
    # Drawdata()
    Drawclass()