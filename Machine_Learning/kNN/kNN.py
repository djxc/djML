# -*- coding:utf-8  -*-
from numpy import *
import numpy as np
import operator
from os import listdir

def createDataSet():
    group=np.array(([1.0,1.1],[1.0,1.0],[0,0],[0,0.1]))
    lables=['A','A','B','B']
    return group,lables

def classify0(inX, dataSet, labels, k):
    """
        KNN,k-最近邻算法：简单，但是运算量大。通过计算输入数据与每个样本数据之间的几何距离，选择前k个距离最近的，
    k个最近的标签比重最大的即为预测输入样本的标签。
        这里需要输入1.inX为需要预测的数据; 2.dataSet样本数据; 3.labels样本数据的标签; 4.k为距离最近的前k个样本
    """
    dataSetSize = dataSet.shape[0]      # 计算样本个数
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet      # 使用tile生成与dataSet相同个数的矩阵，然后减去dataSet得到，每个样本与输入数据差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)     # sum()加上axis=1即为每一行求和
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()        # 将distances按照大小对下标排序
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     # 将labels排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()     #截取掉回车符
        listFromLine = line.split('\t')     #截取掉空格
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#检查分类正确率
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("the classifier came back with: %d, the real answer is: %d") % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f") % (errorCount/float(numTestVecs))
    print(errorCount)

def classfyPerson():
    resultList=['not at all','in small doses','in lage doses']
    percenTats=float(raw_input("percentage of time spent playing video games?"))
    ffMile=float(raw_input("frequent flier miles earned per year?"))
    iceCream=float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr=array([ffMile,percenTats,iceCream])
    classifiResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifiResult-1])

#将图像转换为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d") % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d") % errorCount
    print("\nthe total error rate is: %f") % (errorCount/float(mTest))