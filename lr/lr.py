# -*- coding:utf-8 -*-
# @Time : 2019/7/22 16:43
# @Author : naihai

import numpy as np
import math
from utils import accuracy_score, normalize

"""
LR 二项分类
"""


class LogisticRegression:

    def __init__(self, learning_rate=.1, reg=0.1, multi_class=False):
        self.reg = reg
        self.multi_class = multi_class
        self.learning_rate = learning_rate
        self.params = None  # w 向量

    def _initialize_weights(self, n_features):
        """
        初始化权重向量
        :param n_features:
        :return:
        """
        limit = 1.0 / math.sqrt(n_features)
        self.params = np.random.uniform(-limit, limit, (n_features,))

    def _loss_and_gradient(self, X, y):
        """
        使用GD计算梯度与损失
        :param X:
        :param y:
        :return:
        """
        m_samples, n_features = X.shape
        y_pred = sigmoid(X.dot(self.params))
        gradient = X.T.dot(y_pred - y)
        gradient = 1.0 / m_samples * gradient + self.reg * self.params  # 均值 正则化
        # 计算loss
        loss = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))  # 经验风险损失
        # 结构风险损失
        loss = - 1.0 / m_samples * loss + 0.5 * self.reg * self.params * self.params
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
        self._initialize_weights(n_features)

        classes_ = np.unique(y)  # 类的所有取值
        n_classes = len(classes_)
        if n_classes < 2:
            raise ValueError("At least Two classes")

        if self.multi_class:

            pass
        else:
            for i in range(n_iterations):
                loss, gradient = self._loss_and_gradient(X, y)
                self.params -= self.learning_rate * gradient
                if i % 100 == 0:
                    print("loss of iteration {0} is {1}: ".format(i, loss))

    def predict(self, X):
        y_pred = sigmoid(X.dot(self.params))
        return np.round(y_pred).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / (np.sum(ex, axis=-1, keepdims=True))


if __name__ == '__main__':
    """
    下面使用iris作为二分类数据集使用 
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    # 只是用 1 2 类
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression(0.1, 0.01)
    clf.fit(X_train, y_train, 10000)

    print(accuracy_score(y_test, clf.predict(X_test)))
