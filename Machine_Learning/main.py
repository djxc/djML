# -*- coding: utf-8 -*-
import numpy as np
from Perceptron import Perceptron
from aboutData import preprocessData
import matplotlib.pyplot as plt
from kNN import kNN
from aboutData import transformData
    
def testPerceptron(X, y):
    percep = Perceptron(15, 0.01)
#    percep.fit(X, y)
    percep.fitAdaline(X, y)
#    percep.fitAdalineShuffle(X, y)
    percep.showCost()
#    print(percep.predict(X_))

def usingKNN():
    group, labels = kNN.createDataSet()
    predict = kNN.classify0([0, 0], group, labels, 3)
    print(predict)


def usingKNN1():
    tfData = transformData.TransformData()
    datingDataMat, datingLabels = tfData.file2matrix('Data/datingTestSet2.txt')
    print(datingDataMat)
    
    KNNTest(preproData(datingDataMat), datingLabels)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()
    
def preproData(datingDataMat):
    od = preprocessData.OperateData()
    normMat, ranges, minVals = od.autoNorm(datingDataMat)   
    return normMat
    
#检查分类正确率
def KNNTest(normMat, datingLabels):
    hoRatio = 0.10      #hold out 10%
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN.classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))
    print(errorCount)

if __name__ == "__main__":
#    od = preprocessData.OperateData()
#    X, y = od.getData()
#    X_std = od.standardization(X)
##    od.showData(X, y)
#    testPerceptron(X_std, y)
   usingKNN1()
