# -*- coding: utf-8 -*-
# @author djxc
# @date 2020-01-31
# 使用tensorflow识别手写数字，mnist

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, datasets

def load_data():
    '''加载数据，对原始数据以及标签进行初步调整'''
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    # 原始数据为灰度图值在0-255，这里将其缩放在-1 ~ 1之间
    x = 2 * tf.convert_to_tensor(x, dtype=tf.float32)/255. - 1     
    x = np.expand_dims(x, axis=3)
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    print(x.shape, y.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(320)
    return train_dataset

def create_model():
    '''
    1、创建模型，添加不同的layer，需要设置每一层的维度以及激活函数
    2、创建优化函数，设置学习率
    '''
    model = keras.Sequential([ 
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)])
    optimizer = optimizers.SGD(learning_rate=0.001)
    return model, optimizer 

def create_cnn_model():
    """"""
    model = keras.Sequential(
        [
            layers.Conv2D(kernel_size=3, filters=16, activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Conv2D(kernel_size=3, filters=32, activation='relu'),
            layers.MaxPool2D(pool_size=2, strides=2),
            layers.Conv2D(kernel_size=3, filters=32, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ]
    )
    optimizer = optimizers.RMSprop(learning_rate=0.001)
    return model, optimizer

def train_model(model: keras.Sequential, train_dataset, optimizer, epoch_num):
    '''
    1、根据循环次数，循环train次数
    2、train_dataset为一个batch，遍历每一数据，利用模型计算输出；
    通过梯度下降模型更新计算参数。
    '''
    for epoch in range(epoch_num):
        for step, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # x = tf.reshape(x, (-1, 28*28))  # 将二维数组拉平 [b, 28, 28] => [b, 784]
                out = model(x)                  # 计算预测输出，[b, 784] => [b, 10]
                # loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]   # Step2. compute loss
                
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, out))
            # Step3. optimize and update w1, w2, w3, b1, b2, b3
            grads = tape.gradient(loss, model.trainable_variables)
            # w' = w - lr * grad
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', loss.numpy())

def diy_trian(x, y_onehot, lr):
    '''通过张量构建网络'''
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))

    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))

    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    x = tf.reshape(x, [-1, 28*28])
    h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    h2 = x@w2 + b2
    h2 = tf.nn.relu(h2)
    out = h2@w3 + b3

    loss = tf.square(y_onehot - out)
    loss = tf.reduce_mean(loss)
    with tf.GradientTape() as tape:
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
        w2.assign_sub(lr*grads[2])
        b2.assign_sub(lr*grads[3])
        w3.assign_sub(lr*grads[4])
        b3.assign_sub(lr*grads[5])

if __name__ == "__main__":
    train_dataset = load_data()
    model, optimizer = create_cnn_model()
    train_model(model, train_dataset, optimizer, 50)