#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:19:27 2019

@author: djxc
"""

import tensorflow as tf
import cv2
import numpy as np

tf.reset_default_graph()
img = cv2.imread('test.png')
img = cv2.resize(img, (1024, 1024))
img = np.array(img).astype(np.float32)
# 增加一个维度[-1,1024,1024,3]
img = img[np.newaxis, ...]
x_input = tf.placeholder(shape=[None, 1024, 1024, 3], dtype=tf.float32)


def conv2D(x, k_num, name, pool=True):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(x, k_num, (3, 3), strides=1, padding='same')
        conv = tf.layers.batch_normalization(conv, training=True)
        conv = tf.nn.relu(conv)

        conv = tf.layers.conv2d(conv, k_num, (3, 3), strides=1, padding='same')
        conv = tf.layers.batch_normalization(conv, training=True)
        conv = tf.nn.relu(conv)
        if pool is False:
            return conv

        maxpool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return conv, maxpool



def upsampling(tensor, size=[2, 2]):
    '''
    针对pooling操作的un-pooling操作
    '''
    h, w, c = tensor.get_shape().as_list()[1:]
    h_multi, w_multi = size
    h = h * h_multi
    w = w * w_multi

    return tf.image.resize_nearest_neighbor(tensor, size=(h, w))


def up_concat(tensorA, tensorB, name='up_concat'):
    '''
    将对应的卷积层的特征图和输入融合
    '''
    upsampling_A = upsampling(tensorA)
    return tf.concat([upsampling_A, tensorB], axis=-1, name=name)


# 归一化
x_input = (x_input - 127.5) / 127.5

# conv1 1024*1024*3 --> 1024*1024*8,512*512*8
conv1, pool1 = conv2D(x_input, 8, pool=True, name='conv1')
# 512*512*8 --> 512*512*16,256*256*16
conv2, pool2 = conv2D(pool1, 16, pool=True, name='conv2')
# 256*256*16 -->256*256*32,128*128*32
conv3, pool3 = conv2D(pool2, 32, pool=True, name='conv3')
# 128*128*32 -->128*128*64,64*64*64
conv4, pool4 = conv2D(pool3, 64, pool=True, name='conv4')

# 64*64*64 -->64*64*128
conv5 = conv2D(pool4, 68, pool=False, name='conv5')

# 64*64*128+128*128*64 -->128*128*192
up6 = up_concat(conv5, conv4, name='up6')
# 128*128*192 -->128*128*64
conv6 = conv2D(up6, k_num=64, pool=False, name='conv6')

# 128*128*64+256*256*32 -->256*256*96
up7 = up_concat(conv6, conv3, name='up7')
# 256*256*96  -->256*256*32
conv7 = conv2D(up7, k_num=32, pool=False, name='conv7')

# 256*256*32+512*512*16 --> 512*512*48
up8 = up_concat(conv7, conv2, name='up8')
# 512*512*48  -->512*512*16
conv8 = conv2D(up8, k_num=16, pool=False, name='conv8')

# 512*512*16+1024*1024*8 --> 1024*1024*24
up9 = up_concat(conv8, conv1, name='up9')
# 1024*1024*24  -->1024*1024*8
conv9 = conv2D(up9, k_num=8, pool=False, name='conv9')

# 最后一层
final = tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(final, feed_dict={x_input: img})
    result = res[0, ...]

    for i in range(1024):
        for j in range(1024):
            result[i, j, 0] = int(result[i, j, 0] * 255)
            if result[i, j, 0] > 255:
                result[i, j, 0] = 255
    cv2.imwrite('res.jpg', result)