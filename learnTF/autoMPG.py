# -*- coding: utf-8 -*-
# @author djxc
# @date 2020-01-31
# 使用tensorflow测试auto mpg数据

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import pandas as pd
from autoMPGModel import Network

def get_data():
    '''获取数据并初步处理数据'''
    dataset_path = keras.utils.get_file("auto-mpg.data", 
        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
        'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
        na_values = "?", comment='\t',
        sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()
    print(dataset.head())
    print(dataset.isna().sum())
    dataset = dataset.dropna()
    print(dataset.isna().sum())
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0

    print(dataset.tail())
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    train_stats = train_dataset.describe()
    print(train_stats)
    train_stats = train_stats.transpose()
    print(train_stats)
    norm_train_data = norm_data(train_dataset, train_stats)
    norm_test_data = norm_data(test_dataset, train_stats)
    train_db = tf.data.Dataset.from_tensor_slices((
        norm_train_data.values,
        train_labels.values
    ))
    test_db = tf.data.Dataset.from_tensor_slices((
        norm_test_data.values,
        test_labels.values
    ))
    train_db = train_db.shuffle(100).batch(32)
    return train_db, test_db

def norm_data(x, train_stats):
    '''标准化数据'''
    return (x - train_stats['mean'])/train_stats['std']


def train_model(train_db, test_db):
    ''''''
    model = Network()
    model.build(input_shape = (4, 9))
    model.summary()
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    for epoch in range(100):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.reduce_mean(tf.keras.losses.mse(y, out))
                mae_loss = tf.reduce_mean(tf.keras.losses.mae(y, out))
            if step % 10 == 0:
                print(epoch, step, float(loss), float(mae_loss))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    for step, (x, y) in enumerate(test_db):
        print(x, y)
        out = model(x)
        # loss = tf.reduce_mean(tf.keras.losses.mse(y, out))
        # mae_loss = tf.reduce_mean(tf.keras.losses.mae(y, out))
        # if step % 10 == 0:
        #     print(epoch, step, float(loss), float(mae_loss))


if __name__ == "__main__":
    train_db, test_db = get_data()
    train_model(train_db, test_db)