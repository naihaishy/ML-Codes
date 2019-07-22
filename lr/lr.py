# -*- coding:utf-8 -*-
# @Time : 2019/7/22 16:43
# @Author : naihai

import numpy as np
import math


class LogisticRegression:

    def __init__(self, learning_rate=.1):
        self.learning_rate = learning_rate
        self.params = None  # w 向量

    def _initialize_weights(self, n_features):
        limit = math.sqrt(1 / n_features)
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
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    clf = LogisticRegression(0.1)
    clf.fit(X, y, 1000)

    print(clf.predict(X[:2, :]))

    pass
