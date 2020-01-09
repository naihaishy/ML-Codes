# -*- coding:utf-8 -*-
# @Time : 2019/8/1 16:57
# @Author : naihai

""""
FM 实现 用于二分类 与 回归
"""
import numpy as np
from utils import sigmoid

REGRESSION = 0
CLASSIFICATION = 1


class FM(object):

    def __init__(self, task, lr, reg, K):
        self.task = task  # 0 for regression 1 for classification
        self.K = K
        self.lr = lr  # learning_rate
        self.reg = reg  # list
        self.wo = 0
        self.W = None  # n向量
        self.V = None  # nxk矩阵

        self.max_value = None
        self.min_value = None

        self.norm_l2 = True
        assert (task == 0 or task == 1)

    def _initialize(self, n_features, labels=None):
        """
        初始化参数
        :param n_features: n 特征维度
        :return:
        """

        self.W = np.random.normal(0.0, 0.1, (n_features, 1))
        self.V = np.random.normal(0.0, 0.1, (n_features, self.K))

        self.W = np.asmatrix(self.W)
        self.V = np.asmatrix(self.V)

        if self.task == REGRESSION:
            self.max_value = np.max(labels)
            self.min_value = np.min(labels)

    def _predict_instance(self, x):
        """
        预测单个样本
        :param x:  one sample vector or 1xd np.matrix
        :return: float predicted value
        """
        inter_1 = x * self.V
        inter_2 = np.multiply(x, x) * np.multiply(self.V, self.V)  # multiply对应元素相乘
        # 完成交叉项,xi*vi*xi*vi - xi^2*vi^2
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2)

        y_pred = self.wo + (x * self.W)[0, 0] + 0.5 * interaction
        if self.task == 0:
            y_pred = max(self.min_value, min(y_pred, self.max_value))
            # print(y_pred)

        return y_pred

    def _calculate_gradient_and_update_weights(self, X, y):
        """
        :param X: NxD  m_samples, n_features np.matrix
        :param y: N
        :return:
        """
        m_samples, n_features = X.shape

        losses = 0.0
        losses_reg = 0.0

        for m in range(m_samples):
            y_pred = self._predict_instance(X[m])
            inter_1 = X[m] * self.V
            loss = 0.0
            # calculate delta
            delta = 0.0
            if self.task == REGRESSION:
                delta = y_pred - y[m]
            elif self.task == CLASSIFICATION:
                "classification"
                delta = (sigmoid(y[m] * y_pred) - 1) * y[m]

            # update weights
            self.wo -= self.lr * (delta + 2 * self.reg[0] * self.wo)
            for i in range(n_features):
                self.W[i, 0] -= self.lr * (delta * X[m, i] + 2 * self.reg[1] * self.W[i, 0])

            for i in range(n_features):
                for f in range(self.K):
                    gradient_v = delta * (X[m, i] * inter_1[0, f] - self.V[i, f] * (X[m, i] * X[m, i]))
                    self.V[i, f] -= self.lr * (gradient_v + 2 * self.reg[2] * self.V[i, f])

            if self.task == REGRESSION:
                loss = 0.5 * (y_pred - y[m]) * (y_pred - y[m])
            elif self.task == CLASSIFICATION:
                loss = -np.log(sigmoid(y[m] * y_pred))
            losses += loss
            losses_reg += loss + self.reg[0] * abs(self.wo) + \
                          self.reg[0] * np.sum(np.abs(self.W)) + \
                          self.reg[0] * np.sum(np.abs(self.V))

        return losses / m_samples, losses_reg / m_samples

    def fit(self, X, y, n_iterations, verbose=False):
        """
        训练模型
        :param verbose:
        :param X: Mxn
        :param y: M
        :param n_iterations:
        :return:
        """
        m_samples, n_features = X.shape

        if not isinstance(X, np.matrix):
            X = np.asmatrix(X, dtype=np.double)

        self._initialize(n_features, y)

        shuffle_indices = np.arange(X.shape[0])

        for epoch in range(n_iterations):
            # shuffle data
            np.random.shuffle(shuffle_indices)
            X = X[shuffle_indices]
            y = y[shuffle_indices]
            losses, losses_reg = self._calculate_gradient_and_update_weights(X, y)
            if verbose:
                print("epoch {0} loss : {1}, reg loss : {2} ".format(epoch, losses, losses_reg))

    def predict(self, X):
        m_samples, n_features = X.shape

        if not isinstance(X, np.matrix):
            X = np.asmatrix(X, dtype=np.double)

        result = []
        for m in range(m_samples):
            y_pred = self._predict_instance(X[m])

            if self.task == REGRESSION:
                result.append(y_pred)
            elif self.task == CLASSIFICATION:
                if sigmoid(y_pred) >= 0.5:
                    result.append(1)
                else:
                    result.append(-1)

        return result
