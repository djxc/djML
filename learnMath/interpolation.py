
import math
import numpy as np
import sympy as sym
from matplotlib import pyplot as plt


def create_data():
    """生成样本数据"""
    x = [0, 2, 4, 6, 8, 10]
    y = [math.sin(i) for i in x]
    return np.array(x), np.array(y)

def show_point_data(x, y):
    #定义坐标轴
    fig=plt.figure()
    ax1=plt.axes()
    # 基于ax1变量绘制三维图
    #设置坐标轴
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.scatter(x, y)
    plt.show()#显示图像

def show_line_data(x, y, x_inter, y_inter, title='interpolation'):
    #定义坐标轴
    fig=plt.figure()
    ax1=plt.axes()
    # 基于ax1变量绘制三维图
    #设置坐标轴
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.plot(x, y, 'o', label='Point')
    ax1.plot(x_inter, y_inter, label="inter")
    plt.legend()
    plt.title(title)
    plt.show()#显示图像


def line_interpolation(x, y, inter_num=1):
    """线性插值
        1、首先求出x的范围
        2、遍历x范围，默认间隔为1
    """
    x_inter = []
    y_inter = []

    x_min = x[0]
    x_max = x[-1]
    x_p = 0
    for x1 in range(x_min, x_max - 1, inter_num):
        x_inter.append(x1)
        if x1 not in x:
            print(x_p, y[x_p], y[x_p + 1])
            y1 = (y[x_p] + y[x_p + 1]) / 2
        else:
            x_p = x.index(x1)
            y1 = y[x_p]
        print(x1, y1)
        y_inter.append(y1)    
    show_line_data(x, y, x_inter, y_inter)



def fill_table(points_x, points_y):
    n = len(points_x)  # 要插值的点数

    # 定义表格
    ki = np.arange(0, n, 1) # 0-n 取值，间隔为1，ki为【0,1，。。。，n-1】，应该是作为编号了
    # print("ki",ki)
    table = np.concatenate(([ki], [points_x], [points_y]), axis=0)
    # [[0.   1.   2.   3.  ]
    #  [2.2  5.8  4.2  4.5 ]
    #  [4.12 8.42 7.25 7.85]]   把ki，x，y上的数据合并，按照行，第一行是ki，第二行是x，第三行是y

    table = np.transpose(table) # 对数据进行转置，第一行变为第一列，第二行变为第二列，类推
    # [[0.   2.2  4.12]
    #  [1.   5.8  8.42]
    #  [2.   4.2  7.25]
    #  [3.   4.5  7.85]]

    dfinite = np.zeros(shape=(n, n), dtype=float)   # n行n列的0

    table = np.concatenate((table, dfinite), axis=1)    # table, dfinite按照列合并
    # [[0.   2.2  4.12 0.   0.   0.   0.  ]
    #  [1.   5.8  8.42 0.   0.   0.   0.  ]
    #  [2.   4.2  7.25 0.   0.   0.   0.  ]
    #  [3.   4.5  7.85 0.   0.   0.   0.  ]]

    # Calcul la tabla
    [n, m] = np.shape(table)    # 获取table的形状 4 行7列
    diagonal = n - 1    # 对角中值的数
    for j in range(3, m): # 因为从标号为3的列才需要把计算的值插进去
        step = j - 2    # 区之间隔，为了找到x的值
        for i in range(diagonal):
            denominator = (points_x[i + step] - points_x[i])    # 分母
            numerator = table[i + 1, j - 1] - table[i, j - 1]   # 分子计算
            table[i, j] = numerator / denominator   # 获得查分值，保存
        diagonal = diagonal - 1
    # print(table)
    # (8.42-4.12)/(5.8-2.2) = 1.1944444444444444,剩下的这个表格也就能看懂了
    # [[ 0.          2.2         4.12        1.19444444 -0.23159722 -0.32363666 0.        ]
    #  [ 1.          5.8         8.42        0.73125    -0.97596154  0.         0.        ]
    #  [ 2.          4.2         7.25        2.          0.          0.         0.        ]
    #  [ 3.          4.5         7.85        0.          0.          0.         0.        ]]
    return table


def create_polinomio(tabla, n, points_y, points_x):
    diference_divid = tabla[0, 3:]  # [ 1.19444444 -0.23159722 -0.32363666  0.        ] 获取第一行有用的查分信息
    # print(diference_divid)

    x = sym.Symbol('x')
    polynomial = points_y[0]    # 开始构建多项式，第一项就是f（x0）
    for j in range(1, n, 1):
        factor = diference_divid[j - 1] # 分别获取差分中的元素
        term = 1
        for k in range(0, j):
            term = term * (x - points_x[k]) # 构建x
        polynomial = polynomial + term * factor # 把这一项加入到多项式中

    simple_polynomial = polynomial.expand()
    # 把多项式展开-0.323636659234485*x**3 + 3.7167700204385*x**2 - 11.9565732998885*x + 15.8813775083612

    numerical_polynomial = sym.lambdify(x, simple_polynomial)
    return numerical_polynomial


def newton_method(points_x, points_y):
    table = fill_table(points_x, points_y)  # 创建表格
    polynomial = create_polinomio(table, len(points_x), points_y, points_x)  # 创建插值多项式

    # 图形点
    muestras = 101
    min_interval = np.min(points_x)
    max_interval = np.max(points_x)
    points_interval = np.linspace(min_interval, max_interval, muestras)  # x的插值点
    evaluate_points = polynomial(points_interval)  # 根据x算出y
    show_line_data(points_x, points_y, points_interval, evaluate_points)


if __name__ == "__main__":
    xi, fi = create_data()
    # print(x, y)
    # # show_data(x, y)
    # line_interpolation(x, y)

    newton_method(xi, fi) 