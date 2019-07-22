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

    def __init__(self, learning_rate=.1):
        self.learning_rate = learning_rate
        self.params = None  # w 向量

    def _initialize_weights(self, n_features):
        limit = 1.0 / math.sqrt(n_features)
        self.params = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations):
        """
        :param X: [m_samples, n_features]
        :param y:
        :param n_iterations:
        :return:
        """
        m_samples, n_features = X.shape
        self._initialize_weights(n_features)

        for i in range(n_iterations):
            print("iteration {0}".format(i))
            y_pred = sigmoid(X.dot(self.params))
            w_grad = X.T.dot(y_pred - y)
            self.params -= self.learning_rate * w_grad

    def predict(self, X):
        y_pred = sigmoid(X.dot(self.params))
        return np.round(y_pred).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    clf = LogisticRegression(0.01)
    clf.fit(X_train, y_train, 1000)

    print(accuracy_score(y_test, clf.predict(X_test)))
