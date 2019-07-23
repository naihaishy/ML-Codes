# -*- coding:utf-8 -*-
# @Time : 2019/7/22 16:43
# @Author : naihai

import numpy as np
import math
from utils import accuracy_score, normalize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

"""
LR Sigmoid二项分类 与Softmax 多项分类
"""


class LogisticRegression:

    def __init__(self, learning_rate=.1, reg=0.1, multi_class=False):
        self.reg = reg
        self.multi_class = multi_class
        self.learning_rate = learning_rate
        self.W = None  # w 向量 N 多分类时为矩阵 CxD

    def _initialize_weights(self, n_features, n_classes=2):
        """
        初始化权重向量
        :param n_features:
        :return:
        """
        limit = 1.0 / math.sqrt(n_features)
        if self.multi_class:
            self.W = np.random.uniform(-limit, limit, (n_classes, n_features))
        else:
            self.W = np.random.uniform(-limit, limit, (n_features,))

    def _loss_and_gradient_sigmoid(self, X, y):
        """
        二分类 sigmoid
        使用GD计算梯度与损失
        :param X: NxD  m_samples, n_features
        :param y: N
        :return:
        """
        m_samples, n_features = X.shape
        y_pred = sigmoid(X.dot(self.W))
        # 计算梯度
        gradient = - 1.0 / m_samples * X.T.dot(y - y_pred) + self.reg * self.W
        # 计算loss
        loss = - 1.0 / m_samples * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))  # 经验风险损失
        loss = loss + 0.5 * self.reg * np.sum(self.W * self.W)  # 结构风险损失
        return loss, gradient

    def _loss_and_gradient_softmax(self, X, y):
        """
        多分类 softmax
        :param X:
        :param y:
        :return:
        """
        m_samples, n_features = X.shape

        scores = X.dot(self.W.T)  # 矩阵NxC  每个样本在C个类上的预测分数
        # 经过Softmax转换为概率
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # 防止溢出 减去最大值 因此全部小于等于0
        scores_exp_sum = np.sum(scores_exp, axis=-1, keepdims=True)  # 每个样本在C个类上的预测概率之和 归一化

        # 每个样本真实的标签对应的softmax概率
        y_label_softmax = scores_exp[range(m_samples), y].reshape((scores_exp.shape[0], 1)) / scores_exp_sum
        # 计算损失
        loss = - 1.0 / m_samples * np.sum(np.log(y_label_softmax))
        loss = loss + 0.5 * self.reg * np.sum(self.W * self.W)

        y_pred = scores_exp / scores_exp_sum

        gradient = -1.0 / m_samples * X.T.dot(y_label_softmax - y_pred).T + self.reg * self.W
        return loss, gradient

    def fit(self, X, y, n_iterations):
        """
        训练模型
        :param X: [m_samples, n_features]
        :param y: [m_samples, ]
        :param n_iterations:
        :return:
        """
        m_samples, n_features = X.shape

        classes_ = np.unique(y)  # 类的所有取值
        n_classes = len(classes_)
        if n_classes < 2:
            raise ValueError("At least Two classes")

        self._initialize_weights(n_features, n_classes)

        if self.multi_class:
            for i in range(n_iterations):
                loss, gradient = self._loss_and_gradient_softmax(X, y)
                # loss, gradient = loss_grad_softmax_vectorized(self.W, X.T, y, self.reg)
                self.W -= self.learning_rate * gradient
                if i % 100 == 0:
                    print("loss of iteration {0} is {1}: ".format(i, loss))
        else:
            for i in range(n_iterations):
                loss, gradient = self._loss_and_gradient_sigmoid(X, y)
                self.W -= self.learning_rate * gradient
                if i % 100 == 0:
                    print("loss of iteration {0} is {1}: ".format(i, loss))

    def predict(self, X):
        if self.multi_class:
            y_pred = softmax(X.dot(self.W.T))
            return np.argmax(y_pred, axis=1)
        else:
            y_pred = sigmoid(X.dot(self.W))
            return np.round(y_pred).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / (np.sum(ex, axis=-1, keepdims=True))


def binary_classify():
    data = load_digits(2)
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=12)

    clf = LogisticRegression(0.01, 0.01, multi_class=False)
    clf.fit(X_train, y_train, 10000)

    print(accuracy_score(y_test, clf.predict(X_test)))


def multi_classify():
    data = load_digits(10)
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=12)

    clf = LogisticRegression(0.1, 0.01, multi_class=True)
    clf.fit(X_train, y_train, 10000)

    print(accuracy_score(y_test, clf.predict(X_test)))


if __name__ == '__main__':
    multi_classify()
