# -*- coding:utf-8 -*-
# @Time : 2020/1/9 19:16
# @Author : naihai
import argparse
import logging
import math
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from factorization_machine import FMTF


def build_data():
    data = pd.read_csv("E:\Scala\projects\Recommend\data\house_price_train.txt", sep='\t', header=None)
    X = data[data.columns[1:]].values
    y = data[0].values

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=12)
    print(X_train_.shape)

    # 预处理
    scaler = StandardScaler()
    scaler.fit(X)
    X_train_ = scaler.transform(X_train_)
    X_test_ = scaler.transform(X_test_)

    # 验证集
    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X_train_, y_train_, test_size=0.2, random_state=15)
    return X_train_, y_train_, X_valid_, y_valid_, X_test_, y_test_


if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''

    X_train, y_train, X_valid, y_valid, X_test, y_test = build_data()

    # initialize FM model
    m_samples, n_features = X_train.shape
    n_factors = 1000
    lr = 0.01
    model = FMTF(0, lr, n_factors, [0.1, 0.1, 0.1], n_features, decay_steps=100, decay_rate=0.8)
    model.build_graph()  # build graph for model

    batch_size = 32
    n_batches = int(m_samples / batch_size) + 1
    n_epochs = 10000

    display_step = 50
    checkpoint_step = 100

    data_indices = np.arange(m_samples)

    saver = tf.train.Saver(max_to_keep=5)

    best_valid_loss = math.inf
    last_valid_improvement = 0
    require_improvement = 200

    with tf.Session() as sess:
        # TODO: with every epoch, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        for epoch in range(n_epochs):
            # shuffle data
            np.random.shuffle(data_indices)
            feed_dict = {model.X: X_train[data_indices],
                         model.y: y_train[data_indices]}
            loss, reg_loss, train_op, global_step = \
                sess.run([model.loss, model.reg_loss, model.train_op, model.global_step], feed_dict=feed_dict)

            valid_feed_dict = {model.X: X_valid, model.y: y_valid}
            valid_loss, valid_reg_loss = sess.run([model.loss, model.reg_loss], feed_dict=valid_feed_dict)
            print("epoch valid {0} loss : {1} , reg_loss : {2} ".format(epoch, valid_loss, valid_reg_loss))

            if epoch % 1000 == 0:
                model.lr += 0.01
                print(model.lr)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                last_valid_improvement = epoch

            if epoch - last_valid_improvement > require_improvement:
                # 过了require_improvement迭代次都没有降低loss
                print("no improvement found , stopping optimization")
                break

            if (epoch + 1) % display_step == 0:
                pass
                # print("epoch train {0} loss : {1} , reg_loss : {2} ".format(epoch, loss, reg_loss))
            if (epoch + 1) % checkpoint_step == 0:
                saver.save(sess, "./checkpoints/fm", global_step=global_step)

        # Test model
        feed_dict = {model.X: X_test, model.y: y_test}
        print(X_test.shape)
        pres, loss, reg_loss = sess.run([model.y_pred, model.loss, model.reg_loss], feed_dict=feed_dict)
        pres = np.asarray(pres).reshape([-1, 1])
        print("test data loss : {0} , reg_loss : {1} ".format(loss, reg_loss))
        print(mean_squared_error(y_test, pres))
