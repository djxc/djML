# -*- coding: utf-8 -*-
import numpy as np

class TransformData:    
    """数据的转换"""
    def __init__(self):    
        self.age = 28
        
    def file2matrix(self, filename):
        """将文件转化为矩阵"""
        fr = open(filename)
        numberOfLines = len(fr.readlines())         #get the number of lines in the file
        returnMat = np.zeros((numberOfLines,3))        #prepare matrix to return
        classLabelVector = []                       #prepare labels return
        fr = open(filename)
        index = 0
        for line in fr.readlines():
            line = line.strip()     # 截取掉空格
            listFromLine = line.split('\t')     # 按照tab键进行分割
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector        
