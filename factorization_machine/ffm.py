# -*- coding:utf-8 -*-
# @Time : 2019/8/2 14:30
# @Author : naihai
""""
FFM 实现 用于二分类
"""
import numpy as np
from utils import sigmoid


class FFM(object):
    def __init__(self, learning_rate, K, F):
        self.F = F  # field num
        self.K = K  # latent factor num
        self.learning_rate = learning_rate
        self.wo = 0
        self.W = None  # n向量
        self.V = None

        self.featureToField = None  # feature到Filed的映射

    def _initialize(self, n_features):
        """
        初始化参数
        :param n_features: n 特征维度
        :return:
        """
        self.W = np.random.normal(0.0, 0.1, n_features)
        self.V = np.random.normal(0.0, 0.1, (n_features, self.F, self.K))

    def _loss_and_gradient_sigmoid(self, X, y):
        """
        二分类 sigmoid
        使用GD计算梯度与损失
        :param X: NxD  m_samples, n_features np.ndarray
        :param y: N
        :return:
        """
        m_samples, n_features = X.shape

        loss = 0.0

        for m in range(m_samples):
            interaction = 0.0
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction += np.dot(self.V[i, self.featureToField[j]], self.V[j, self.featureToField[i]])

            y_pred = self.wo + np.dot(self.W, X[m]) + interaction

            delta = (sigmoid(y[m] * y_pred) - 1) * y[m]

            # 更新参数
            self.wo = self.wo - self.learning_rate * delta
            for i in range(n_features):
                if X[m, i] != 0:
                    self.W[i] = self.W[i] - self.learning_rate * delta * X[m, i]
                    for j in range(i + 1, n_features):
                        self.V[i, self.featureToField[j]] -= self.learning_rate * delta * self.V[
                            j, self.featureToField[i]] * X[m, i] * X[m, j]
                        self.V[j, self.featureToField[i]] -= self.learning_rate * delta * self.V[
                            i, self.featureToField[j]] * X[m, i] * X[m, j]
            # 计算loss
            loss += -np.log(sigmoid(y[m] * y_pred))
        print(loss/m_samples)

    def fit(self, X, y, n_iterations):
        """
        训练模型
        :param X: Mxn
        :param y: M
        :param n_iterations:
        :return:
        """
        m_samples, n_features = X.shape
        self._initialize(n_features)
        for _ in range(n_iterations):
            self._loss_and_gradient_sigmoid(X, y)

    def predict(self, X):
        m_samples, n_features = X.shape

        if not isinstance(X, np.matrix):
            X = np.asmatrix(X, dtype=np.double)
