
import math
import numpy as np
from numpy import ndarray
import sympy as sym
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import numpy.polynomial.polynomial as poly
from scipy.interpolate import lagrange


def create_data():
    """生成样本数据"""
    x = [i for i in range(0, 30, 2)]
    y = [math.sin(i * 0.5) for i in x]
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


class DInterpolation:
    def __init__(self, x: ndarray, f: ndarray, inter_num=1) -> None:
        self.x = x
        self.f = f
        self.inter_num = inter_num
        self.x_new = np.arange(self.x[0], self.x[-1] + self.inter_num, self.inter_num)

        self.xf_map = self.__create_xf_map()

    def __create_xf_map(self):
        """"""
        x_y_map = {}
        for i, xi in enumerate(self.x):
            x_y_map[xi] = self.f[i]
        return x_y_map

    
    def __find_next_key_value(self, key):
        """在map中查找最小的大于该key的key以及value"""
        for keyi in self.xf_map:
            if keyi > key:
                return keyi
    
    def __find_last_key_value(self, key):
        """在map中查找最小的大于该key的key以及value"""
        last_key = 0
        for keyi in self.xf_map:
            if keyi > key:
                return last_key
            last_key = keyi


    def line_interpolation(self):
        """线性插值
            1、建立x与y的对应关系
            2、首先求出x的范围
            3、遍历x范围，默认间隔为1
        """

        x_inter = []
        y_inter = []
        last_y1 = self.f[0]
        next_y1 = None
        for x1 in self.x_new:
            x_inter.append(x1)
            if x1 not in self.xf_map:    
                if next_y1 is None:           
                    next_x1 = self.__find_next_key_value(x1)
                    next_y1 = self.xf_map[next_x1]
                y1 = last_y1 + (next_y1 - last_y1) / ((next_x1 - x1) / self.inter_num)
                last_y1 = y1
            else:
                y1 = self.xf_map[x1]
                last_y1 = y1
                next_y1 = None
            y_inter.append(y1)    
        return x_inter, y_inter
    
    def cubic_interpolation(self):
        """三次样条插值"""
        # use bc_type = 'natural' adds the constraints as we described above
        f = CubicSpline(self.x, self.f, bc_type='natural')

        y_new = f(self.x_new)
        return self.x_new, y_new
    
    def lagrange_interpolation(self):
        """拉格朗日插值"""
        f = lagrange(self.x, self.f)
        y_new = f(self.x_new)
        return self.x_new, y_new
    
    def polynomial_interpolation(self, power_num=2):
        """多项式插值
            1、根据给出的点构造多项式，多项式的次数需要小于样本点
        """        
        simple_num = self.f.shape[0]
        assert power_num <= simple_num, "多项式次数需要小于等于样本数"
        p = np.poly1d(np.polyfit(self.x, self.f, power_num))       # 多项式拟合
        print(p) 
        x_inter = []
        y_inter = []       

        for x1 in self.x_new:
            x_inter.append(x1) 
            y_inter.append(p(x1))     
        return x_inter, y_inter
    
    def nowton_interpolation(self):
        '''
        evaluate the newton polynomial 
        at x
        '''
        a_s = self.__divided_diff(self.x, self.f)[0, :]
        x_min = self.x[0]
        x_max = self.x[-1]
        n = len(self.x) - 1 
        p = a_s[n]
        x = np.arange(x_min, x_max + 1, self.inter_num)
        for k in range(1,n+1):
            p = a_s[n-k] + (x - self.x[n-k])*p
        return x, p

   

    def __divided_diff(self, x, y):
        '''
        function to calculate the divided
        differences table
        '''
        n = len(y)
        coef = np.zeros([n, n])
        # the first column is y
        coef[:,0] = y
        
        for j in range(1,n):
            for i in range(n-j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
        return coef



if __name__ == "__main__":
    # xi = np.arange(0, 8)
    # fi = np.array([4350755.6, 4349116, 4347471, 4345821, 4344167, 4342507, 4340842, 4339171])
    # yi = np.array([3302338, 3308915, 3315488, 3322056, 3328622, 3335182, 3341740.0, 3348293.8])
    # zi = np.array([4358314, 4354978, 4351637, 4348292, 4344941, 4341584, 4338224, 4334858])
    xi, fi = create_data()
    # show_point_data(xi, zi)
    interpolation = DInterpolation(xi, fi, 0.1)
    x_inter, y_inter = interpolation.line_interpolation()
    # # x_inter, y_inter = interpolation.cubic_interpolation()
    # x_inter, y_inter = interpolation.lagrange_interpolation()
    # # # x_inter, y_inter = interpolation.polynomial_interpolation(power_num=10)
    # # # x_inter, y_inter = interpolation.nowton_interpolation()

    show_line_data(xi, fi, x_inter, y_inter)