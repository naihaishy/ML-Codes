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
            results[m] = y_pred
        elif task == CLASSIFICATION:
            y_pred = 1 if sigmoid(y_pred) >= 0.5 else 0
            results[m] = int(y_pred)

    return [ele for ele in results]



def sigmoid(double x):
    return 1.0 / (1.0 + exp(-x))
