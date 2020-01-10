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

    def __init__(self, task, learning_rate, K, reg_l1, reg_l2, batch_size, n_features, n_classes=0):
        self.task = task
        self.lr = learning_rate
        self.reg_l1 = reg_l1  # list b W, V
        self.reg_l2 = reg_l2
        self.n_features = n_features
        self.K = K

        self.batch_size = batch_size

        self.X = None
        self.y = None
        self.keep_prob = None
        self.y_pred = None
        self.y_pred_prob = None
        self.loss = None

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
        self.keep_prob = tf.placeholder('float32')

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
            linear_terms = tf.add(tf.reduce_sum(tf.multiply(W, self.X)), w0)

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
        optimizer = tf.train.GradientDescentOptimizer(self.lr)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()
