# -*- coding:utf-8 -*-
# @Time : 2019/8/1 16:57
# @Author : naihai

""""
FM 实现 用于二分类
"""
import numpy as np
from utils import sigmoid


class FM(object):

    def __init__(self, learning_rate, reg, K):
        self.K = K
        self.learning_rate = learning_rate
        self.reg = reg
        self.wo = 0
        self.W = None  # n向量
        self.V = None  # nxk矩阵

    def _initialize(self, n_features):
        """
        初始化参数
        :param n_features: n 特征维度
        :return:
        """
        self.W = np.random.normal(0.0, 0.1, (n_features, 1))
        self.V = np.random.normal(0.0, 0.1, (n_features, self.K))

        self.W = np.asmatrix(self.W)
        self.V = np.asmatrix(self.V)

    def _loss_and_gradient_sigmoid(self, X, y):
        """
        二分类 sigmoid
        使用GD计算梯度与损失
        :param X: NxD  m_samples, n_features np.matrix
        :param y: N
        :return:
        """
        m_samples, n_features = X.shape

        loss = 0.0

        for m in range(m_samples):
            inter_1 = X[m] * self.V
            inter_2 = np.multiply(X[m], X[m]) * np.multiply(self.V, self.V)  # multiply对应元素相乘
            # 完成交叉项,xi*vi*xi*vi - xi^2*vi^2
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.0

            y_pred = self.wo + (X[m] * self.W)[0, 0] + 0.5 * interaction
            delta = (sigmoid(y[m] * y_pred) - 1) * y[m]

            self.wo = self.wo - self.learning_rate * delta
            for i in range(n_features):
                if X[m, i] != 0:
                    self.W[i, 0] = self.W[i, 0] - self.learning_rate * delta * X[m, i]
                    for f in range(self.K):
                        self.V[i, f] = self.V[i, f] - self.learning_rate * delta * (
                                X[m, i] * inter_1[0, f] - self.V[i, f] * (X[m, i] * X[m, i]))

            loss += -np.log(sigmoid(y[m] * y_pred))
        print(loss / m_samples)

    def fit(self, X, y, n_iterations):
        """
        训练模型
        :param X: Mxn
        :param y: M
        :param n_iterations:
        :return:
        """
        m_samples, n_features = X.shape

        if not isinstance(X, np.matrix):
            X = np.asmatrix(X, dtype=np.double)

        self._initialize(n_features)
        for _ in range(n_iterations):
            self._loss_and_gradient_sigmoid(X, y)

    def predict(self, X):
        m_samples, n_features = X.shape

        if not isinstance(X, np.matrix):
            X = np.asmatrix(X, dtype=np.double)

        result = []
        for m in range(m_samples):
            inter_1 = X[m] * self.V
            inter_2 = np.multiply(X[m], X[m]) * np.multiply(self.V, self.V)  # multiply对应元素相乘
            # 完成交叉项,xi*vi*xi*vi - xi^2*vi^2
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.0

            y_pred = self.wo + (X[m] * self.W)[0, 0] + 0.5 * interaction

            if sigmoid(y_pred) >= 0.5:
                result.append(1)
            else:
                result.append(-1)
        return result
