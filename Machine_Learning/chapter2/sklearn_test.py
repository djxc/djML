# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from show_tool import myShow
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
"""通过数据收集，标准化处理，然后进行算法学习，评价算法分类效果
    1.逻辑回归算法
    2.（核）支持向量机
    3.决策树算法
    4.随机从林算法
    5.k最邻算法"""


def process_data():
    """对数据集进行预处理
    1.加载数据集
    2.划分为训练集合测试集
    3.进行标准化"""
    X_train, X_test, y_train, y_test = get_data()
    """将数据进行标准化，减少因单位不同引起的误差"""
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test


def get_data():
    """获取数据"""
    iris = datasets.load_iris()  # 加载数据
    X = iris.data[:, [2, 3]]  # 特征数据
    y = iris.target  # 标签label
    print(X[5:11, :])
    print(np.unique(y))
    # 将特征数据和标签按比例随机分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def draw_result(cla):
    """画出数据的类别分区，显示数据"""
    myshow = myShow()
    myshow.plot_decision_regions_new(X=X_combined_std, y=y_combined,
                                     classifier=cla, test_idx=range(105, 150))
    show_plt()


def show_plt():
    """面板的坐标代表值，以及图例"""
    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend(loc='upper left')
    plt.show()


def show_qulity(y_predict):
    """显示算法错分的个数，评价算法的精确度"""
    print('Misclassified sample:%d' % (y_predict != y_test).sum())
    print('Accuracy : %.2f' % accuracy_score(y_test, y_predict))


def create_data():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)     # 随机产生200行数据，每行有两列。以0为中心
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)    # 判断这二百个数，每个是否大于零，大于零设为true，否则为false
    y_xor = np.where(y_xor, 1, -1)      # true为1，false为-1
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')   # y_xor为1的生成点
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')
    plt.ylim(-3.0)
    plt.legend()
    plt.show()
    return X_xor, y_xor


def perceptron_test():
    ppn = Perceptron(max_iter=60, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_predict = ppn.predict(X_test_std)
    show_qulity(y_predict)
    draw_result(ppn)


def logistic_regression_test():
    """逻辑回归算法"""
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    draw_result(lr)
    lr_pre = lr.predict(X_test_std)
    
    print('Accuracy : %.2f' % accuracy_score(y_test, lr_pre))


def SVM_test():
    """支持向量机算法"""
    svm = SVC(max_iter=60, kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    y_predict = svm.predict(X_test_std)
    show_qulity(y_predict)
    draw_result(svm)


def kernel_SVM_test():
    """带核的支持向量机算法，使用rbf核，将二维非线性的数据投影到三维空间中，可以进行线性划分"""
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
    svm.fit(X_xor, y_xor)
    myshow = myShow()
    myshow.plot_decision_regions_new(X_xor, y_xor, classifier=svm)
    plt.legend(loc='upper left')
    plt.show()


def kernel_SVM_iris():
    svm = SVC(kernel='rbf', gamma=0.2, C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    y_predict = svm.predict(X_test_std)
    show_qulity(y_predict)
    draw_result(svm)


def decision_tree():
    """决策树算法"""
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    X_train, X_test, y_train, y_test = get_data()
    tree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    myshow = myShow()
    myshow.plot_decision_regions_new(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    y_predict = tree.predict(X_test)
    export_graphviz(tree, out_file='F:/tree.dot', feature_names=['petal length', 'petal width'])
    show_qulity(y_predict)
    show_plt()


def random_forest():
    """随机从林算法"""
    X_train, X_test, y_train, y_test = get_data()
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    myshow = myShow()
    myshow.plot_decision_regions_new(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    show_plt()


def knn_test():
    """k最邻算法"""
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    myshow = myShow()
    myshow.plot_decision_regions_new(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))
    show_plt()

if __name__ == "__main__":
    X_train_std, X_test_std, y_train, y_test = process_data()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # logistic_regression_test()
    # SVM_test()
    # X_xor, y_xor = create_data()
    # kernel_SVM_test()
    # SVM_test()
    # kernel_SVM_iris()
    # decision_tree()
    # random_forest()
    knn_test()
