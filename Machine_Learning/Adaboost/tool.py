
from numpy import *
import boost


d=mat(ones((5,1))/5)
dataMat,classLabel=boost.loadSimpData()
# boost.buildStump(dataMat,classLabel,d)
classfierArr= boost.adaBoostTrainDS(dataMat,classLabel,30)
print boost.adaClassify([0,0],classfierArr)