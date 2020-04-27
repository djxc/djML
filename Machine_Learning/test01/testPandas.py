import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#创建一维数组，可以为数组每个元素生成序号
s=pd.Series([1,4,8,14,np.nan,20])
print(s)

#创建日期列，第二个参数为列的长度
dates=pd.date_range('20171117',periods=6)
print(dates)

#通过数组创建二维表格，第一个属性为填充的值，第二个为左侧的行号，第三个为表头列名
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)

#通过对象字典创建二维表格
df2=pd.DataFrame({'A':1,
                  'B':pd.Timestamp('20171117'),
                  'C':pd.Series(1,index=range(4),dtype='float32'),
                  'D':np.array([3]*4,dtype=int),
                  'E':pd.Categorical(["test","dj","xc","dj"]),
                  'F':'foo'})
print(df2)


#对于DateFram的描述操作
def descriptDF():
    # 输出DateFram 的前几行
    print(df.head(3))
    # 输出DateFram 的后几行
    print(df.tail(3))
    # 输出DateFram行的索引
    print(df.index)
    # 输出DateFram列的名称
    print(df.columns)
    # 输出DateFram里面的值
    print(df.values)
    #输出DateFram的描述，每列的个数、平均值、最小值等等
    print(df.describe())
    #矩阵的转置
    print(df.T)
    #通过坐标轴排序
    print(df.sort_index(axis=1,ascending=False))
    #根据给定的某一列的值进行排序
    print(df.sort_values(by='A'))

#选择DateFram的某些元素
def selectDF():
    #选择一列数据，返回为series类型
    print(df['A'])
    #选择某一行到某一行之间的数值
    print(df[1:3])
    #通过位置选择
    print(df.iloc[2])
    #查询‘B’列中值大于0的
    print(df[df.B>0])

#对于DateFram进行操作
def operateDF():
    #计算DateFram的每一列的平均值
    print(df.mean())

    # 计算DateFram的每一行的平均值
    print(df.mean(1))

#读取和写出数据
def writeread():
    # df.to_csv('foo.csv')
    r1=pd.read_csv('foo.csv')
    print(r1)
    #写出到excel表
    df.to_excel('foo1.xlsx', sheet_name='foo')
    # r2=pd.read_excel('foo.xlsx', 'foo', index_col=None, na_values=['NA'])
    # print(r2)

#类型分类
def categorical():
    #创建DataFrame表，包含两列ID，raw_grade
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
    print(df)
    df["grade"]=df["raw_grade"].astype("category")#将原始数据转换为可以分类的数据类型,并为DateFram添加新的列
    print(df["grade"])
    df["grade"].cat.categories=["very good","good","very bad"]#重新设定标签
    print(df.sort_values(by="grade"))
    #统计每种标签的个数
    print(df.groupby("grade").size())

#测试绘图工具
def testPoltting():
    #创建一千个数的随机数数列，为一维数组,并将其显示出来
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()
    # ts.plot()
    # plt.legend(loc='best')#图例的位置
    # plt.show()

    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns = ['A', 'B', 'C', 'D'])
    df = df.cumsum()
    plt.figure();
    df.plot();
    plt.legend(loc='best')
    plt.show()

#通过某一列的进行group，相同值得合并
def testgroup():
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                       'C': np.random.randn(8),
                       'D': np.random.randn(8)})
    print(df.groupby('A').sum())#将'A'列进行合并，其他的列进行求和

#两个矩阵合并，merge扩展列，append扩展行
def testmerge():
    left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
    right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
    print(pd.merge(left,right,on='key'))

# selectDF()
# operateDF()
# writeread()
# categorical()
# testPoltting()
# testgroup()
testmerge()