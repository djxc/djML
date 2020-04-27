#-！- coding:utf-8 -!-
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from aboutData import OperateData
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from show_tool import myShow
from sklearn.metrics import accuracy_score

def testPerceptron():
    """测试perceptron算法，并计算错误率，显示分类结果"""
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    showResult(ppn)

def testLogisticRegression(X_train_std, y_train):
    """测试逻辑回归算法"""
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    lr_pre = lr.predict(X_test_std)
    print('Accuracy : %.2f' % accuracy_score(y_test, lr_pre))
    showResult(lr)
#    print(lr.predict_proba(X_test_std[0,:]))
def testSVM():
    """支持向量机算法"""
    svm = SVC(max_iter=60, kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    y_predict = svm.predict(X_test_std)
    showQulity(y_predict)
    showResult(svm)
    
def logistic_regression_test():
    """逻辑回归算法"""
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    showResult(lr)
    lr_pre = lr.predict(X_test_std)    
    showQulity(lr_pre)

def showQulity(y_predict):
    """显示算法错分的个数，评价算法的精确度"""
    print('Misclassified sample:%d' % (y_predict != y_test).sum())
    print('Accuracy : %.2f' % accuracy_score(y_test, y_predict))

def showResult(cla):
    myshow = myShow()
    myshow.plot_decision_regions_new(X=X_combined_std, y=y_combined,
                                     classifier=cla, test_idx=range(105, 150))
    myshow.showPlt()
    
def getData():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    type=np.unique(y) #获取y的不同的类型
#    print(type)
    # 将数据进行分为测试数据与训练数据，测试数据比例为0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 将数据进行标准化
    od = OperateData()
    X_train_std, X_test_std = od.standardizationSL(X_train, X_test)
    return X_train_std, X_test_std, y_train, y_test




if __name__ == '__main__':
    X_train_std, X_test_std, y_train, y_test = getData()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
#    testPerceptron()
    testLogisticRegression(X_train_std, y_train)
   
    # DrawData()