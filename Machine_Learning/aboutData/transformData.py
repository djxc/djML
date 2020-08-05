# -*- coding: utf-8 -*-
import numpy as np


class TransformData:
    """数据的转换"""

    def __init__(self):
        self.age = 28

    def file2matrix1(self, filename):
        """将文件转化为矩阵"""
        fr = open(filename)
        # get the number of lines in the file
        numberOfLines = len(fr.readlines())
        returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return
        classLabelVector = []  # prepare labels return
        fr = open(filename)
        index = 0
        for line in fr.readlines():
            line = line.strip()     # 截取掉空格
            listFromLine = line.split('\t')     # 按照tab键进行分割
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector

    def file2matrix(self, filename):
        '''读取文件，将其转换为矩阵  
            @param filename 需要读取的文件  
            @return returnMat返回的特征矩阵  
            @return classLabelVector 返回的标签数组向量。
        '''
        fr = open(filename)
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)  # get the number of lines in the file
        # 构建元素全部为0的numberOLines行，3列的，存储标训练数据据矩阵
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []  # 存储标签数组
        index = 0
        for line in arrayOLines:
            line = line.strip()  # 截取掉回车符
            listFromLine = line.split('\t')  # 按tab进行分割
            returnMat[index, :] = listFromLine[0:3]     # 前三个数据为特征数据
            classLabelVector.append(int(listFromLine[-1]))  # 最后一个数据为标签数据
            index += 1
        return returnMat, classLabelVector
