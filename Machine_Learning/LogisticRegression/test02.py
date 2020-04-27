#-！- coding:utf-8 -!-
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import myData


def testLR():

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    myData.plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))

    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

def testSVM():
    # svm = SVC(kernel='linear', C=1.0, random_state=0)#这个方法为线性分类
    svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)#这个方法为非线性分类
    svm.fit(X_train_std, y_train)

    myData.plot_decision_regions(X_combined_std,y_combined, classifier = svm,test_idx = range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

if  __name__ == '__main__':
    X_train_std, X_test_std, y_train, y_test = myData.getData()
    print("-----------------------------")
    print(X_test_std[0,:])
    print("-----------------------------")
    print(X_test_std[:,0:2]) 
#    X_combined_std = np.vstack((X_train_std, X_test_std))
#    y_combined = np.hstack((y_train, y_test))
#    # testLR()
#    testSVM()