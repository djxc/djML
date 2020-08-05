# 用来可视化数据
# @author djxc
# @date 2020-02-03
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt

class Show():
    def __init__(self):
        pass

    def show_3d(self, data, title='none'):
        '''显示三维数'''
        fig = plt.figure(title)
        ax = fig.gca(projection='3d')
        ax.plot_surface(data[0], data[1], data[2])
        ax.view_init(60, -30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    def show_moons(self, X, y, plot_name, file_name=None, dark=False):
        ''''''
        if(dark):
            plt.style.use('dark_background')
        plt.figure(figsize=(16, 12))
        axes = plt.gca()
        axes.set(xlabel="x", ylabel="y")
        plt.title(plot_name, fontsize=30)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(right=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral,
            edgecolors='none')
        plt.show()
