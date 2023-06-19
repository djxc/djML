

import numpy as np
from matplotlib import pyplot as plt
 

def draw_points():
    #定义坐标轴
    fig=plt.figure()
    ax1=plt.axes(projection='3d')
    # 基于ax1变量绘制三维图
    #设置xyz方向的变量（空间曲线）
    z=np.linspace(0,13,1000)#在1~13之间等间隔取1000个点
    x=5*np.sin(z)
    y=5*np.cos(z)
    
    #设置xyz方向的变量（散点图）
    zd=13*np.random.random(100)
    xd=5*np.sin(zd)
    yd=5*np.cos(zd)
    
    #设置坐标轴
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.scatter(xd,yd,zd,cmap='Blues') #绘制散点图
    ax1.plot3D(x,y,z,'gray')#绘制空间曲线
    plt.show()#显示图像

def draw_suface():
    #定义新坐标轴
    fig=plt.figure()
    ax3=plt.axes(projection='3d')
    
    #定义三维数据
    xx=np.arange(-5,5,0.2)
    yy=np.arange(-5,5,0.2)
    
    #生成网格点坐标矩阵,对x和y数据执行网格化
    X,Y=np.meshgrid(xx,yy)
    
    #计算z轴数据
    Z=np.sin(X)+np.cos(Y)
    
    #绘图
    #函数plot_surface期望其输入结构为一个规则的二维网格
    ax3.plot_surface(X,Y,Z, rstride = 1, cstride = 1,cmap='rainbow') #cmap是颜色映射表
    plt.title("3D")
    plt.show()

def contour():
    #定义新坐标轴
    fig4 = plt.figure()
    ax4 = plt.axes(projection='3d')
    
    #生成三维数据
    xx = np.arange(-5,5,0.1)
    yy = np.arange(-5,5,0.1)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(np.sqrt(X**2+Y**2))
    
    #作图
    ax4.plot_surface(X,Y,Z,alpha=0.8,cmap='winter')     #生成表面， alpha 用于控制透明度
    ax4.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面,offset表示离视角轴0点的距离
    ax4.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
    ax4.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
    #ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数
    
    #设定显示范围
    ax4.set_xlabel('X')
    ax4.set_xlim(-6, 4)  #设置坐标轴范围显示投影
    ax4.set_ylabel('Y')
    ax4.set_ylim(-4, 6)
    ax4.set_zlabel('Z')
    ax4.set_zlim(-3, 3)
    
    plt.show()

def lsq_method():
    """最小二乘法"""
    # 拟合曲线
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy as sp
    from scipy.optimize import leastsq

    # 样本数据
    # 身高数据
    Xi = np.array([162, 165, 159, 173, 157, 175, 161, 164, 172, 158])
    # 体重数据
    Yi = np.array([48, 64, 53, 66, 52, 68, 50, 52, 64, 49])


    # 需要拟合的函数func（）指定函数的形状
    def func(p, x):
        k, b = p
        return k*x + b


    # 定义偏差函数，x，y为数组中对应Xi,Yi的值
    def error(p, x, y):
        return func(p, x) - y


    # 设置k，b的初始值，可以任意设定，经过实验，发现p0的值会影响cost的值：Para[1]
    p0 = [1, 20]

    # 把error函数中除了p0以外的参数打包到args中,leastsq()为最小二乘法函数
    Para = leastsq(error, p0, args=(Xi, Yi))
    # 读取结果
    k, b = Para[0]
    print('k=', k, 'b=', b)

    # 画样本点
    plt.figure(figsize=(8, 6))
    plt.scatter(Xi, Yi, color='red', label='Sample data', linewidth=2)

    # 画拟合直线
    x = np.linspace(150, 180, 80)
    y = k * x + b

    # 绘制拟合曲线
    plt.plot(x, y, color='blue', label='Fitting Curve', linewidth=2)
    plt.legend()  # 绘制图例

    plt.xlabel('Height:cm', fontproperties='simHei', fontsize=12)
    plt.ylabel('Weight:Kg', fontproperties='simHei', fontsize=12)

    plt.show()


if __name__ == "__main__":
    # draw_suface()
    # contour()
    lsq_method()