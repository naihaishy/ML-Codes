"""
Factorization machines in cython
"""
from math import log1p

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, pow
from cpython cimport bool

DEF REGRESSION = 0
DEF CLASSIFICATION = 1

cdef class FactorizationMachine(object):
    """垃圾实现"""
    # 定义成员变量 C type
    cdef public double w0
    cdef public np.ndarray W  # # n向量
    cdef public np.ndarray V  # nxk矩阵
    cdef public int n_factors
    cdef public int n_features
    cdef public int task
    cdef public double lr
    cdef public double reg
    cdef public double max_value
    cdef public double min_value

    def __init__(self,
                 double w0,
                 np.ndarray[np.double_t, ndim=1, mode='c'] W,
                 np.ndarray[np.double_t, ndim=2, mode='c'] V,
                 int n_factors,
                 int n_features,
                 int task,
                 double lr,
                 double reg,
                 double max_value,
                 double min_value):
        self.task = task
        self.lr = lr
        self.reg = reg
        self.n_factors = n_factors
        self.n_features = n_features

        self.w0 = w0
        self.W = W  # n向量
        self.V = V  # nxk矩阵
        self.max_value = max_value
        self.min_value = min_value

    cdef _predict_instance(self, np.ndarray[np.double_t, ndim=1, mode='c'] x):
        """ 预测单个样本 """
        # 将实例对象的变量映射到本地变量
        cdef double w0 = self.w0
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] W = self.W
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = self.V

        cdef int i, j, f

        cdef double interaction = 0.0
        cdef double linear = 0.0
        cdef double y_pred = 0.0

        cdef double inter_1, inter_2
        n_factors = self.n_factors
        n_features = self.n_features

        for f in range(n_factors):
            inter_1 = 0.0
            inter_2 = 0.0

            for i in range(n_features):
                inter_1 += x[i] * V[i, f]
                inter_2 += V[i, f] * V[i, f] * x[i] * x[i]

            interaction += inter_1 * inter_1 - inter_2

        for i in range(n_features):
            linear += W[i] * x[i]

        y_pred += w0 + linear + interaction
        return y_pred

    cdef _gradient_and_update_instance(self, np.ndarray x, double y):
        # 映射到本地变量
        cdef double w0 = self.w0
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] W = self.W
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = self.V
        cdef double max_value = self.max_value
        cdef double min_value = self.min_value
        cdef int n_factors = self.n_factors
        cdef int n_features = self.n_features
        cdef int task = self.task
        cdef double lr = self.lr
        cdef double reg = self.reg
        cdef double y_pred = 0.0
        cdef double delta = 0.0

        cdef int i, j, f,

        cdef double loss = 0.0

        cdef np.ndarray[np.double_t, ndim=1] inter_sum_1 = np.zeros(self.n_factors, np.double)

        y_pred = self._predict_instance(x)

        if task == REGRESSION:
            y_pred = min(max_value, y_pred)
            y_pred = max(min_value, y_pred)

        for f in range(n_factors):
            inter_sum_1[f] = 0.0
            for j in range(n_features):
                inter_sum_1[f] += x[j] * V[j, f]

        # calculate delta
        delta = 0.0
        if task == REGRESSION:
            "regression"
            delta = y_pred - y
        elif task == CLASSIFICATION:
            "classification"
            delta = (sigmoid(y * y_pred) - 1) * y

        # update weights
        w0 = w0 - lr * delta
        for i in range(n_features):
            if x[i] != 0:
                W[i] = W[i] - lr * delta * x[i]
                for f in range(n_factors):
                    V[i, f] = V[i, f] - lr * delta * (x[i] * inter_sum_1[f] - V[i, f] * (x[i] * x[i]))

        if task == REGRESSION:
            "regression"
            loss = 0.5 * (y_pred - y) * (y_pred - y)
        elif task == CLASSIFICATION:
            "classification"
            loss = log(sigmoid(y * y_pred))
        self.w0 = w0
        self.W = W
        self.V = V
        return loss

    cdef _gradient_and_update(self, np.ndarray[np.double_t, ndim=2, mode='c'] X,
                              np.ndarray[np.double_t, ndim=1, mode='c'] y):

        cdef int m_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        assert n_features == self.n_features

        cdef double losses = 0.0
        cdef np.ndarray[np.double_t, ndim=1] inter_sum_1 = np.zeros(self.n_factors, np.double)
        cdef int m

        for m in range(m_samples):
            losses += self._gradient_and_update_instance(X[m], y[m])
        print(losses / m_samples)

    def fit(self, np.ndarray X, np.ndarray y, int n_iterations):
        """
        训练模型
        :param X: Mxn
        :param y: M
        :param n_iterations:
        :return:
        """
        for _ in range(n_iterations):
            self._gradient_and_update(X, y)

    def predict(self, X):

        cdef int m_samples = X.shape[0]
        cdef int n_features = X.shape[1]

        assert n_features == self.n_features

        result = []
        for m in range(m_samples):
            y_pred = self._predict_instance(X[m])

            if self.task == REGRESSION:
                y_pred = min(self.max_value, y_pred)
                y_pred = max(self.min_value, y_pred)

            if self.task == REGRESSION:
                "regression"
                result.append(y_pred)
            elif self.task == CLASSIFICATION:
                "classification"
                if sigmoid(y_pred) >= 0.5:
                    result.append(1)
                else:
                    result.append(0)

        return result

cdef sigmoid(double x):
    return 1.0 / (1.0 + exp(-x))
