# -*- coding:utf-8 -*-
import bayes
import re
import feedparser

def function01():
    listOPOST,listclass=bayes.loadDataSet()
    myVocabList=bayes.createVocabList(listOPOST)
    print myVocabList

    print bayes.setOfWords2Vec(myVocabList,listOPOST[0])

    bayes.testingNB()
    mySent="My name is dujie,I am from China."
    print mySent.split()
    regEx=re.compile('\\W*')
    print regEx.split(mySent)
    bayes.spamTest()


def function02():
    ny = feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
    sf= feedparser.parse('https://newyork.craigslist.org/stp/index.rss')
    covabList,pSF,pNY=bayes.localWords(ny,sf)
    covabList1, pSF1, pNY1 = bayes.localWords(ny, sf)
    bayes.getTopWords(ny,sf)

if __name__ == '__main__':
    function02()