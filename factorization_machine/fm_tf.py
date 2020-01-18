# -*- coding:utf-8 -*-
# @Time : 2019/8/2 14:40
# @Author : naihai

"""
FM 的TensorFlow实现
Factorization Machine
Regression Classification
"""
import argparse
import logging
import pickle

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.training import saver

REGRESSION = 0
CLASSIFICATION = 1


class FMTF(object):

    def __init__(self, task, learning_rate, K, reg, n_features, decay_steps=100, decay_rate=0.96, n_classes=0):
        self.task = task
        self.lr = learning_rate
        self.reg = reg  # list b W, V
        self.n_features = n_features
        self.K = K

        self.X = None
        self.y = None
        self.y_pred = None
        self.y_pred_prob = None
        self.loss = None
        self.reg_loss = None  # 正则化参数 结构loss

        self.n_classes = n_classes

        self.correct_prediction = None
        self.accuracy = None

        self.global_step = None
        self.train_op = None

        assert (self.task == REGRESSION or self.task == CLASSIFICATION)
        if self.task == CLASSIFICATION:
            assert (self.n_classes > 0)

    def add_placeholder(self):
        self.X = tf.placeholder('float32', [None, self.n_features])
        self.y = tf.placeholder('float32', [None, ])

    def inference(self):
        with tf.variable_scope("linear_layer"):
            if self.task == REGRESSION:
                w0 = tf.get_variable("w0", shape=[1, ],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
                W = tf.get_variable("W", shape=[self.n_features, ],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            elif self.task == CLASSIFICATION:
                w0 = tf.get_variable("w0", shape=[self.n_classes, ],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
                W = tf.get_variable("W", shape=[self.n_features, self.n_classes],
                                    initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

            linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, self.X), axis=1, keepdims=True))

        with tf.variable_scope("interaction_layer"):
            V = tf.get_variable("V", shape=[self.n_features, self.K],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))

            inter_1 = tf.pow(tf.matmul(self.X, V), 2)
            inter_2 = tf.matmul(tf.pow(self.X, 2), tf.pow(V, 2))

            interaction_terms = tf.multiply(0.5, tf.reduce_mean(tf.subtract(inter_1, inter_2), 1, keepdims=True))

        self.y_pred = tf.add(linear_terms, interaction_terms)

        if self.task == CLASSIFICATION:
            self.y_pred_prob = tf.nn.softmax(self.y_pred)

    def add_loss(self):
        if self.task == REGRESSION:
            mse_loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_pred)))
            self.loss = mse_loss
        elif self.task == CLASSIFICATION:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred)
            cross_entropy_loss = tf.reduce_mean(cross_entropy)
            self.loss = cross_entropy_loss

        with tf.variable_scope("linear_layer", reuse=True):
            w0 = tf.get_variable("w0")
            W = tf.get_variable("W")
            self.reg_loss = tf.add(self.reg[0] * tf.reduce_sum(tf.square(w0)),
                                   self.reg[1] * tf.reduce_sum(tf.square(W)))  # L2损失
        with tf.variable_scope("interaction_layer", reuse=True):
            V = tf.get_variable("V")
            self.reg_loss = tf.add(self.reg_loss, self.reg[2] * tf.reduce_sum(tf.square(V)))  # L2损失
        self.loss = self.loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        if self.task == CLASSIFICATION:
            self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_pred, 1), tf.int64), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            # add summary to accuracy
            tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, self.global_step, decay_steps=100, decay_rate=0.96)
        # optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=0.01, l2_regularization_strength=0.01)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer = tf.train.AdagradOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()
