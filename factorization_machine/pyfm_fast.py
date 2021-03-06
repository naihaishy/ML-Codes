# -*- coding:utf-8 -*-
# @Time : 2020/1/7 15:12
# @Author : naihai


import numpy as np
from .fm_fast import fm_fast_train, fm_fast_predict

REGRESSION = 0
CLASSIFICATION = 1


class FactorizationMachine(object):
    def __init__(self, task, lr, reg, K):
        self.task = task  # 0 for regression 1 for classification
        self.K = K
        self.lr = lr  # learning_rate
        self.reg = reg  # list w0 W, V
        self.w0 = 0.0
        self.W = None  # n向量
        self.V = None  # nxk矩阵

        self.max_value = 0.0
        self.min_value = 0.0
        assert (task == 0 or task == 1)

        self.fm_fast = None

    def _initialize(self, n_features, labels=None):
        """ 初始化参数"""
        self.W = np.random.normal(0.0, 0.1, (n_features,))
        self.V = np.random.normal(0.0, 0.1, (n_features, self.K))

        if self.task == REGRESSION:
            self.max_value = np.max(labels)
            self.min_value = np.min(labels)

    def fit(self, X, y, n_iterations, verbose=False):
        m_samples, n_features = X.shape
        self._initialize(n_features, y)

        if not isinstance(X, np.ndarray):
            X = np.asarray(X, np.double, order='C')
        else:
            X = X.copy(order='C')
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, np.double, order='C')
        else:
            y = y.copy(order='C')

        self.w0, self.W, self.V = fm_fast_train(self.w0,
                                                self.W,
                                                self.V,
                                                self.task,
                                                self.lr,
                                                self.reg,
                                                self.max_value,
                                                self.min_value,
                                                X, y, n_iterations, verbose)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, np.double, order='C')
        else:
            X = X.copy(order='C')

        results = fm_fast_predict(self.w0,
                                  self.W,
                                  self.V,
                                  self.task,
                                  self.max_value,
                                  self.min_value,
                                  X)
        if self.task == CLASSIFICATION:
            for i in range(len(results)):
                if results[i] == 0:
                    results[i] = -1
        return results
