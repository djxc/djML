import tree
import treePlotter

# treePlotter.createPlot01()
# mytree= treePlotter.retrieveTree(0)
# treePlotter.createPlot(mytree)
myDate,label=tree.createDataSet()
print myDate
Entroy=tree.calcShannonEnt(myDate)
print Entroy
print tree.splitDataSet(myDate,0,1)
print tree.chooseBestFeatureToSplit(myDate)
print tree.createTree(myDate,label)
print tree.grabTree("classifierStorage.txt")
fr=open("lenses.txt")
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=tree.createTree(lenses,lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)