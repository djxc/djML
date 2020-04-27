import svmMLiA

dataArr,labelArr=svmMLiA.loadDataSet('testSet.txt')
print labelArr

b,alphas=svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)

print b,alphas[alphas>0]