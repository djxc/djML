# -!- encoding:utf-8 -!-
import theano
import theano.tensor as T
import numpy as np
import time
import matplotlib.pyplot as plt

import scipy as sp
import seaborn as sns
import random

from theano import shared



def grad():
    x1 = T.scalar()
    x2 = T.scalar()

    y = x1 * x2
    g = T.grad(y, [x1, x2])

    f = theano.function([x1,x2],y)
    f_prime = theano.function([x1, x2] , g)
    print(f(2,4))
    print(f_prime(2,4))


def grad1():
    A = T.matrix()
    B = T.matrix()

    C = A*B
    D = T.sum(C) # sum为每个元素进行相加
    g = T.grad(D, A)

    y_prime = theano.function([A, B], g)
    A = [[1, 2], [3, 4]]
    B = [[2, 4], [6, 8]]
    print(y_prime(A, B))


def testGPU():
    X = T.matrix(dtype='float32')
    Y = T.matrix(dtype='float32')
    Z = T.dot(X, Y)
    f=theano.function([X, Y], Z)

    x = np.random.randn(10000, 10000).astype(dtype='float32')
    y = np.random.randn(10000, 10000).astype(dtype='float32')
    tStrat = time.time()
    z = f(x, y)
    tEnd = time.time()
    print(tEnd-tStrat)


def testfunction():
    a=T.scalar()
    b=T.matrix()
    c=T.matrix('ha ha ha')
    print(a)
    print(b)
    print(c)


def matrix():
    # 定义输入参数
    a = T.matrix()
    b = T.matrix()
    # 定义输出参数
    c = a*b
    d = T.dot(a, b)
    # 定义函数
    F1 = theano.function([a, b], c)
    F2 = theano.function([a, b], d)

    A = [[1, 2], [3, 4]]
    B = [[2, 4], [6, 8]]
    C = [[1, 2], [3, 4], [5, 6]]
    print(F1(A, B))
    print(F2(C, B))


def sigleneuron():
    x = T.vector()
    w = T.vector()
    b = T.scalar()

    z = T.dot(x, w) + b
    y = 1/(1+T.exp(-z))
    Fun = theano.function([x, w, b], y)
    w = [-1, 1]
    b = 0
    for i in range(100):
        x = [random.random(), random.random()]
        print(x)
        print(Fun(x, w, b))


def sigleneuron1():
    x = T.vector()
    w = theano.shared(np.array([1., 1.]))
    b = theano.shared(0.)

    z = T.dot(w, x) + b
    y = 1/(1+T.exp(-z))
    Fun = theano.function([x], y)

    print(w.get_value())
    w.set_value([0., 0.])

    for i in range(100):
        x = [random.random(), random.random()]
        print(x)
        print(Fun(x))


def sigleneuron2():
    x = T.vector()
    w = theano.shared(np.array([-1., 1.]))
    b = theano.shared(0.)
    y_hat = T.scalar() # 为真实的值

    z = T.dot(w, x) + b
    y = 1/(1+T.exp(-z))  # 预测的值
    cost = T.sum((y-y_hat)**2)   # 计算消耗函数
    dw, db = T.grad(cost, [w, b])  # 求导进行下一步的修改参数

    neuron = theano.function([x], y)
    gradient = theano.function(
        inputs=[x, y_hat],
        outputs=[dw, db]
    )
    x = [1, -1]
    y_hat = 1
    tStrat = time.time()
    for i in range(10000):
        print(neuron(x))
        dw, db = gradient(x, y_hat)
        # 修改参数
        w.set_value(w.get_value() - 0.1*dw)
        b.set_value(b.get_value() - 0.1 * db)

        print(w.get_value(), b.get_value())
    tEnd = time.time()
    print(tEnd - tStrat)


# 用sigmod函数作为激活函数
def sigleneuron3():
    x = T.vector()
    w = theano.shared(np.array([-1., 1.]))
    b = theano.shared(0.)
    y_hat = T.scalar() # 为真实的值

    z = T.dot(w, x) + b
    y = 1/(1+T.exp(-z))  # 预测的值
    cost = T.sum((y-y_hat)**2)   # 计算消耗函数
    dw, db = T.grad(cost, [w, b])  # 求导进行下一步的修改参数

    neuron = theano.function([x], y)
    gradient = theano.function(
        inputs=[x, y_hat],
        updates=[(w, w - 0.1*dw), (b, b - 0.1*db)]
    )
    x = [1, -1]
    y_hat = 1
    tStrat = time.time()
    for i in range(10000):
        print(neuron(x))
        gradient(x, y_hat)
        print(w.get_value(), b.get_value())
    tEnd = time.time()
    print(tEnd - tStrat)


def sigleneuron4():
    x = T.vector()
    w = theano.shared(np.array([-1., 1.]))
    b = theano.shared(0.)
    y_hat = T.scalar()  # 为真实的值

    z = T.dot(w, x) + b
    y = 1/(1+T.exp(-z))  # 预测的值
    cost = T.sum((y-y_hat)**2)   # 计算消耗函数
    dw, db = T.grad(cost, [w, b])  # 求导进行下一步的修改参数

    neuron = theano.function([x], y)
    gradient = theano.function(
        inputs=[x, y_hat],
        updates=MyUpdate([w, b], [dw, db])
    )
    x = [1, -1]
    y_hat = 1
    tStrat = time.time()
    for i in range(10000):
        y_p = neuron(x)
        if abs(y_hat - y_p) < 0.01:
            break
        print(y_p)
        gradient(x, y_hat)
        print(w.get_value(), b.get_value())
    tEnd = time.time()
    print(tEnd - tStrat)


# 用ReLU函数作为激活函数
def sigleneuron5():
    x = T.vector()
    w = theano.shared(np.array([-1., 1.]))
    b = theano.shared(0.)
    y_hat = T.scalar()  # 为真实的值

    z = T.dot(w, x) + b
    y = T.switch(z < 0, -z, 0)  #1/(1+T.exp(-z))  # 预测的值
    cost = T.sum((y-y_hat)**2)   # 计算消耗函数
    dw, db = T.grad(cost, [w, b])  # 求导进行下一步的修改参数

    neuron = theano.function([x], y)
    gradient = theano.function(
        inputs=[x, y_hat],
        updates=MyUpdate([w, b], [dw, db])
    )
    x = [1, -1]
    y_hat = 1
    tStrat = time.time()
    for i in range(10000):
        y_p = neuron(x)
        if abs(y_hat-y_p) < 0.0001:
            break
        print(y_p)
        gradient(x, y_hat)
        print(w.get_value(), b.get_value())
    tEnd = time.time()
    print(tEnd - tStrat)


def XOR():
    x = T.vector()
    w1 = theano.shared(np.random.randn(2))  # 产生两个元素的随机数序列
    b1 = theano.shared(np.random.randn(1))
    w2 = theano.shared(np.random.randn(2))
    b2 = theano.shared(np.random.randn(1))
    w = theano.shared(np.random.randn(2))
    b = theano.shared(np.random.randn(1))

    a1 = 1/(1 + T.exp(-1*(T.dot(w1, x) + b1)))
    a2 = 1 / (1 + T.exp(-1 * (T.dot(w2, x) + b2)))
    y = 1 / (1 + T.exp(-1 * (T.dot(w, [a1, a2]) + b)))

    y_hat = T.scalar()

    cost = -(y_hat*T.log(y) + (1-y_hat)*T.log(1-y)).sum()

    dw, db, dw1, db1, dw2, db2 = T.grad(cost, [w, b, w1, b1, w2, b2])

    gradient = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
        updates=MyUpdate([w, b, w1, b1, w2, b2], [dw, db, dw1, db1, dw2, db2])
    )
    tStrat = time.time()
    for i in range(1000):
        y1, c1 = gradient([0, 0], 0)
        y2, c2 = gradient([0, 1], 1)
        y3, c3 = gradient([1, 0], 1)
        y4, c4 = gradient([1, 1], 0)
        print(c1 + c2 + c3 + c4)
        print(y1, y2, y3, y4)
    tEnd = time.time()
    print(tEnd - tStrat)


def XOR1():
    x = T.vector()
    w1 = theano.shared(np.random.randn(2))  # 产生两个元素的随机数序列
    b1 = theano.shared(np.random.randn(1))
    w2 = theano.shared(np.random.randn(2))
    b2 = theano.shared(np.random.randn(1))
    w = theano.shared(np.random.randn(2))
    b = theano.shared(np.random.randn(1))

    # a1 = 1/(1 + T.exp(-1*(T.dot(w1, x) + b1)))
    # a2 = 1 / (1 + T.exp(-1 * (T.dot(w2, x) + b2)))
    # y = 1 / (1 + T.exp(-1 * (T.dot(w, [a1, a2]) + b)))

    z1 = T.dot(w1, x) + b1
    a1 = T.switch(z1 < 0, -z1, 0)

    z2 = T.dot(w2, x) + b2
    a2 = T.switch(z2 < 0, -z2, 0)

    z3 = T.dot(w, [a1, a2]) + b
    y = T.switch(z3 < 0, -z3, 0)

    y_hat = T.scalar()

    cost = T.sum((y-y_hat)**2) # -(y_hat*T.log(y) + (1-y_hat)*T.log(1-y)).sum()

    dw, db, dw1, db1, dw2, db2 = T.grad(cost, [w, b, w1, b1, w2, b2])

    gradient = theano.function(
        inputs=[x, y_hat],
        outputs=[y, cost],
        updates=MyUpdate([w, b, w1, b1, w2, b2], [dw, db, dw1, db1, dw2, db2])
    )
    tStrat = time.time()
    for i in range(1000):
        y1, c1 = gradient([0, 0], 0)
        y2, c2 = gradient([0, 1], 1)
        y3, c3 = gradient([1, 0], 1)
        y4, c4 = gradient([1, 1], 0)
        print(c1 + c2 + c3 + c4)
        print(y1, y2, y3, y4)
    tEnd = time.time()
    print(tEnd - tStrat)


def dj(X_all, Y_hat_all, data_size, batch_number):
    x = T.matrix('input', dtype='float32')
    w1 = theano.shared(value=np.zeros((1000, 2), dtype=theano.config.floatX), name='W', borrow=True) #T.matrix()
    b1 = theano.shared(T.vector())
    w2 = theano.shared(matrix)
    b2 = theano.shared(T.vector)
    w = theano.shared(matrix)
    b = theano.shared(T.vector)
    y_hat = T.matrix('reference', dtype='float32')

    z1 = T.dot(w, x) + b.dimshuffle(0, 'x')
    a1 = 1/(1+T.exp(-z1))
    z2 = T.dot(w1, a1) + b1.dimshuffle(0, 'x')
    a2 = 1 / (1 + T.exp(-z2))
    z3 = T.dot(w2, a2) + b2.dimshuffle(0, 'x')
    y = 1 / (1 + T.exp(-z3))

    parameters = [w, w1, w2, b, b1, b2]
    cost = T.sum((y - y_hat) ** 2) / batch_number # 计算消耗函数
    gradients = T.grad(cost, parameters)  # 求导进行下一步的修改参数

    train = theano.function(
            inputs=[x, y_hat],
            updates=MyUpdate(parameters, gradients),
            outputs=cost
        )

    test = theano.function(inputs=[x], outputs=y)

    for t in range(100):
        cost = 0
        X_batch, Y_hat_batch = theano.mk_batch(X_all, Y_hat_all, data_size, batch_number )
        for i in range(batch_number):
            cost += train(X_batch[i], Y_hat_batch[i])
        cost /= batch_number
        print(cost)


def drawMiku(X_all, Y_hat_all):
    x = T.vector()
    y_hat = T.scalar()
    w = theano.shared(np.array([-1., 1.]))
    b = theano.shared(0.)

    y = 1 / (1 + T.exp(-1 * (T.dot(x, w) + b)))
    cost = T.sum((y - y_hat) ** 2)  # 计算消耗函数
    dw, db= theano.grad(cost, [w, b])

    gradient = theano.function(inputs=[x, y_hat],
                               updates=MyUpdate([w, b], [dw, db]),
                               outputs=[y, cost])

    for i in range(100):
        for t in range(0, 10):
            y1, c = gradient(X_all[t+i*10], Y_hat_all[t+i*10])
            print(y1, Y_hat_all[t])



def drawMiku1(X_all, Y_hat_all):
    data_size = 100
    batch_number = 100
    x = T.matrix()
    y_hat = T.vector()
    w1 = theano.shared(np.random.randn(2, data_size))
    b1 = theano.shared(np.random.randn(data_size))
    w2 = theano.shared(np.random.randn(data_size))
    b2 = theano.shared(np.random.randn(data_size))
    w3 = theano.shared(np.random.randn(data_size))
    b3 = theano.shared(np.random.randn(data_size))

    a1 = 1 / (1+T.exp(-1*(T.dot(x, w1) + b1)))
    a2 = 1 / (1 + T.exp(-1 * (T.dot(a1, w2) + b2)))
    y = 1 / (1 + T.exp(-1 * (T.dot(a2, w3) + b3)))
    cost = T.sum((y - y_hat) ** 2) / batch_number
    # cost = -(y_hat * T.log(y) + (1 - y_hat) * T.log(1 - y)).sum()
    parameters = [w1, b1, w2, b2, w3, b3]
    dparam = theano.grad(cost, parameters)

    gradient = theano.function(inputs=[x, y_hat],
                               updates=MyUpdate(parameters, dparam),
                               outputs=[y, cost])
    for i in range(100):
        X_batch, Y_hat_batch = mk_batch(X_all, Y_hat_all, data_size, batch_number)
        for t in range(0, batch_number):
            y1, c = gradient(X_batch[t], Y_hat_batch[t])
            print(c)
        print('******************************')


def printdata(X_all, Y_hat_all, data_size, batch_number):
    X_batch, Y_hat_batch = mk_batch(X_all, Y_hat_all, 10, 10)
    w2 = theano.shared(np.random.randn(10, 2))  # 产生两个元素的随机数序列
    y = 1 / (1 + T.exp(-1 * (T.dot(w2, X_batch[0]))))
    cost = T.sum((y - Y_hat_all[0]) ** 2) / batch_number
    print(y)
    # w = []
    # for i in range(4):
    #     w.append(X_all[i])
    print(T.dot(w2, X_batch[0]))

    # for t in range(10):
    #     print(X_batch[t], Y_hat_batch[t])


def mk_batch(X_all, Y_hat_all, data_size, batch_number):
    X_bat = []
    Y_bat = []
    for i in range(batch_number):
        xy = []
        lable = []
        for t in range(data_size):
           xy.append(X_all[t+i*data_size])
           lable.append(Y_hat_all[t+i*data_size])
        X_bat.append(xy)
        Y_bat.append(lable)
    return X_bat, Y_bat


def MyUpdate(paramters, gradients):
    mu = 0.1
    paramter_updates = [(p, p-mu*g) for p, g, in zip(paramters, gradients)]
    return paramter_updates


def f(x):
    return 5 * np.sin(x) + np.sin(5 * x)


def test2():
    blue, _, red, *__ = sns.color_palette()
    SEED = 6533961  # from random.org
    np.random.seed(SEED)
    n = 20
    sigma = 0.2

    xmax = np.pi

    x = sp.stats.uniform.rvs(scale=xmax, size=n)
    y = f(x) + sp.stats.norm.rvs(scale=sigma, size=n)
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_x = np.linspace(0, np.pi, 100)

    ax.plot(plot_x, f(plot_x),
            label='True function');
    ax.scatter(x, y,
               c=red, label='Observations');

    ax.set_xlim(0, xmax);
    ax.legend()
    plt.show()

if __name__ == '__main__':
    # testGPU()
    # testfunction()
    # matrix()
    # grad()
    # grad1()
    sigleneuron()
    # test2()
    # sigleneuron1()
    # sigleneuron2()
    # sigleneuron3()
    # sigleneuron4()
    # XOR()
    # sigleneuron5()
    print(np.ones(10))
