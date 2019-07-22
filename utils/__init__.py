# -*- coding:utf-8 -*-
# @Time : 2019/7/22 17:21
# @Author : naihai
import numpy as np


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def accuracy_score(label, y_pred):
    return np.sum(label == y_pred, axis=0) / len(label)


