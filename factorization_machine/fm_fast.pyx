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

def fm_fast_train(_w0, _W, _V, _task, _lr, _reg,
                  _max_value, _min_value, X, Y, n_iterations, _verbose):
    cdef double w0 = _w0
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] W = _W  # # n向量
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = _V  # nxk矩阵
    cdef int n_factors = _V.shape[1]
    cdef int n_features = _V.shape[0]
    cdef int task = _task
    cdef double lr = _lr
    cdef double reg_w0 = _reg[0]
    cdef double reg_w = _reg[1]
    cdef double reg_v = _reg[2]
    cdef double max_value = _max_value
    cdef double min_value = _min_value
    cdef int n_epochs = n_iterations
    cdef int m_samples = X.shape[0]
    cdef bool verbose = _verbose

    cdef double y_pred, interaction, linear, delta
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] inter_sum_1 = np.zeros(n_factors)
    cdef double inter_1, inter_2
    cdef int i, j, f, m

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] x
    cdef double y
    cdef double losses, losses_reg, loss_reg_w, loss_reg_v, d, gradient_w, gradient_v
    cdef np.ndarray shuffle_indices = np.arange(m_samples)

    for epoch in range(n_epochs):
        # 每次迭代 shuffle data
        np.random.shuffle(shuffle_indices)
        losses = 0.0
        losses_reg = 0.0
        for m in shuffle_indices:
            # 每个样本
            x = X[m]
            y = Y[m]
            inter_sum_1 = np.zeros(n_factors)
            loss_reg_w = 0.0
            loss_reg_v = 0.0

            # predict instance
            y_pred = 0.0
            interaction = 0.0
            linear = 0.0
            for f in range(n_factors):
                inter_1 = 0.0
                inter_2 = 0.0
                inter_sum_1[f] = 0.0

                for i in range(n_features):
                    d = x[i] * V[i, f]
                    inter_1 += d
                    inter_2 += d * d
                    inter_sum_1[f] += d

                interaction += inter_1 * inter_1 - inter_2

            for i in range(n_features):
                linear += W[i] * x[i]

            y_pred += w0 + linear + interaction
            # print(y_pred)

            if task == REGRESSION:
                y_pred = max(min_value, min(max_value, y_pred))

            # calculate delta
            delta = 0.0
            if task == REGRESSION:
                delta = y_pred - y
            elif task == CLASSIFICATION:
                delta = (sigmoid(y * y_pred) - 1) * y

            # update weights
            # update w0
            w0 -= lr * (delta + 2 * reg_w0 * w0)
            # update W
            for i in range(n_features):
                gradient_w = delta * x[i]
                W[i] -= lr * (gradient_w + 2 * reg_w * W[i])
                loss_reg_w += reg_w * abs(W[i])
            # update V
            for i in range(n_features):
                for f in range(n_factors):
                    gradient_v = delta * (x[i] * inter_sum_1[f] - V[i, f] * (x[i] * x[i]))
                    V[i, f] -= lr * (gradient_v + 2 * reg_v * V[i, f])
                    loss_reg_v += reg_v * abs(V[i, f])
                    # print("gradient_v {0}, Vif {1}, delta {2}".format(gradient_v, V[i, f], delta))

            if task == REGRESSION:
                loss = 0.5 * (y_pred - y) * (y_pred - y)
            elif task == CLASSIFICATION:
                loss = -log(sigmoid(y * y_pred))
            losses += loss  # + reg_w0 * w0 + loss_reg_w + loss_reg_v
            losses_reg += loss + reg_w0 * abs(w0) + loss_reg_w + loss_reg_v
            # print("loss {0}, wo loss {1}, W loss {2}, V loss{3} ".format(loss, reg_w0 * w0, loss_reg_w, loss_reg_v))
        # end samples
        if verbose:
            print("epoch {0} loss : {1}, reg loss : {2} ".format(epoch, losses / m_samples, losses_reg / m_samples))
    # end iterations
    return w0, W, V

def fm_fast_predict(_w0, _W, _V, _task, _max_value, _min_value, X):
    cdef double w0 = _w0
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] W = _W  # # n向量
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] V = _V  # nxk矩阵
    cdef int n_factors = _V.shape[1]
    cdef int n_features = _V.shape[0]
    cdef int task = _task
    cdef double max_value = _max_value
    cdef double min_value = _min_value
    cdef int m_samples = X.shape[0]

    cdef double y_pred, interaction, linear, delta
    cdef double inter_1, inter_2
    cdef int i, j, f, m

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] x
    cdef double d
    cdef np.ndarray shuffle_indices = np.arange(m_samples)

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] results = np.zeros(m_samples)

    for m in range(m_samples):
        # 每个样本
        x = X[m]
        loss_reg_w = 0.0
        loss_reg_v = 0.0

        # predict instance
        y_pred = 0.0
        interaction = 0.0
        linear = 0.0
        for f in range(n_factors):
            inter_1 = 0.0
            inter_2 = 0.0

            for i in range(n_features):
                d = x[i] * V[i, f]
                inter_1 += d
                inter_2 += d * d

            interaction += inter_1 * inter_1 - inter_2

        for i in range(n_features):
            linear += W[i] * x[i]

        y_pred += w0 + linear + interaction
        # print(y_pred)

        if task == REGRESSION:
            y_pred = max(min_value, min(max_value, y_pred))
        elif task == CLASSIFICATION:
            y_pred = 1 if sigmoid(y_pred) >= 0.5 else 0

        results[m] = y_pred
    return [ele for ele in results]

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
