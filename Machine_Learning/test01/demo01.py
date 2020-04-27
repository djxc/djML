import numpy as np

#创建矩阵
def createArray():
    # 创建数组,3行2列，类型为int
    x = np.empty([3, 2], dtype=int)
    print(x)

    # 创建数组，以0填充
    x1 = np.zeros([4, 3, 3], dtype=int)
    print(x1)

    x2 = np.ones([2, 3], dtype=[('name', 'S10'), ('age', int)])
    print(x2)

#测试矩阵的样式
def arrayShape():
    student =np.dtype([('name','S20'),('age','i1'),('marks','f4')])
    print(student)
    xc=np.array([('xc',24,85),('dj',25,83)],dtype=student)
    print(xc)

    a1=np.array([[1,2,3],[2,4,3]])
    b=a1.reshape(3,2)

    print("******",b,"******")
    a2=np.array([1,4,2])
    print(a1.shape,a2.shape)

    c1=np.arange(24)
    c2=c1.reshape(2,4,3)
    print(c2)

def testAsarray():
    #使用asarray数组，将数组转换为矩阵
    arr=[1,5,2]
    as1=np.asarray(arr,dtype=float)
    print(as1)

    arr1=(1,6,8)
    as2=np.asarray(arr1)
    print(as2)

#此函数将缓冲区解释为一维数组。 暴露缓冲区接口的任何对象都用作参数来返回ndarray。
def testBuffer():
    str='hello world'
    buffer=np.frombuffer(str,dtype='S1')
    print(buffer)


def testfromiter():
    list=range(6)
    it=iter(list)
    x=np.fromiter(it,dtype=int)
    print(x)

testAsarray()

# testBuffer()

testfromiter()