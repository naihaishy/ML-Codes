# -*- coding:utf-8 -*-
# @Time : 2019/7/23 19:47
# @Author : naihai

"""
损失函数
"""
import numpy as np


class Base(object):
    def loss(self, label, y_pred):
        raise NotImplementedError()

    def gradient(self, label, y_pred):
        raise NotImplementedError()


class CrossEntropy(Base):
    """
    交叉熵损失
    """

    def loss(self, label, y_pred):
        return -label * np.log(y_pred) - (1 - label) * np.log(1 - y_pred)

    def gradient(self, label, y_pred):
        return -(label / y_pred) + (1 - label) / (1 - y_pred)


class SquareLoss(Base):
    """
    平方损失
    """

    def loss(self, label, y_pred):
        return 0.5 * np.power(label - y_pred, 2)

    def gradient(self, label, y_pred):
        return -(label - y_pred)
