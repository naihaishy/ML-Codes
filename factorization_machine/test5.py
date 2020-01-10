# -*- coding:utf-8 -*-
# @Time : 2020/1/9 19:16
# @Author : naihai
import argparse
import logging
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from factorization_machine import FMTF

if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''

    data = pd.read_csv("E:\Scala\projects\Recommend\data\house_price_train.txt", sep='\t', header=None)
    X = data[data.columns[1:]].values
    y = data[0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    mode = "train"
    print_step = 10
    # initialize FM model
    batch_size = 32
    m_samples, n_features = X_train.shape
    n_factors = 100
    lr = 0.0001
    model = FMTF(0, lr, n_factors, 0.01, 0.01, batch_size, n_features)
    # build graph for model
    model.build_graph()

    n_batches = int(m_samples / batch_size) + 1

    data_indices = np.arange(m_samples)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # TODO: with every epoch, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        for epoch in range(100):
            # shuffle data
            np.random.shuffle(data_indices)
            for data_index in np.array_split(data_indices, n_batches):
                X_train_batch = X_train[data_index]
                y_train_batch = y_train[data_index]
                feed_dict = {model.X: X_train_batch,
                             model.y: y_train_batch,
                             model.keep_prob: 1.0}

                loss,  train_op, global_step = sess.run([model.loss, model.train_op, model.global_step], feed_dict=feed_dict)
                # print(data_index)
            print("epoch {0} loss : {1} ".format(epoch, loss))
