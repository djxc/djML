# -*- coding: utf-8 -*-
# @author djxc
# @date 2020-01-31
# 使用tensorflow创建模型

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

class Network(keras.Model):
    ''''''
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1, activation='relu')

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
