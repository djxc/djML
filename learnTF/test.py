# -*- coding: utf-8 -*-
# @author djxc
import sys
import numpy as np
from about_data.create_data import CreateData
from show_data.show import Show

def test():
    args = sys.argv     # 用来获取输入的参数的
    if len(args) == 1:
        print("hello, world")
    elif len(args) > 1:
        print("hello, djxc")

def createData():
    '''随机生成100个数'''
    data = []   # 创建一个list用来存储数据
    for i in range(100):
        x = np.random.uniform(-10., 10.)    # 随机生成-10 ~ 10之间的浮点型数
        eps = np.random.normal(0., 0.01)    # 按照正态分布生成随机误差
        y = 1.477 * x + 0.089 + eps         # 真实值，即为标签
        data.append([x, y])
    data = np.array(data)                   # 将list转化为二维数组
    # print(data)
    return data

def calMSE(b, w, points):
    '''遍历所有的点，计算当前b，w下平均误差'''
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w*x + b))**2
    return totalError/float(len(points))

def step_gradient(b_current, w_current, points, lr):
    '''计算误差函数在每个点上的误差，然后更新w, b,并返回'''
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(len(points)):        # 遍历所有的points计算平均b，w的更新值
        x = points[i, 0]
        y = points[i, 1]
        # 根据误差函数对b,w求导公式，计算b，w变化值
        b_gradient += (2/M) * ((w_current*x + b_current) - y)
        w_gradient += (2/M) * x * ((w_current*x + b_current) - y)
    new_b = b_current - b_gradient*lr
    new_w = w_current - w_gradient*lr
    return [new_b, new_w]

def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    '''梯度下降算法'''
    b = starting_b
    w = starting_w
    for step in range(num_iterations):
        b, w = step_gradient(b, w, points, lr)
        loss = calMSE(b, w, points)
        if step%50 == 0:
            print(f"iteration: {step}, loss: {loss}, w: {w}, b: {b}")
    return [b, w]

def net_moon():
    '''创建网络判别月牙数据'''
    
    pass

if __name__ == "__main__":
    test()
    points = createData()
    [final_b, final_w] = gradient_descent(points, 0, 0, 0.01, 1000)
    loss = calMSE(final_b, final_w, points)
    print(f"Final loss: {loss}, final b: {final_b}, final w: {final_w}")
    data_create = CreateData()
    x, y, z = data_create.himmelblau_data()
    show = Show()
    show.show_3d([x, y, z])

    X, y, X_train, X_test, y_train, y_test = data_create.moons_data()
    show.show_moons(X, y, "test")