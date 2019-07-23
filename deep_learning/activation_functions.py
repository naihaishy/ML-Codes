# -*- coding:utf-8 -*-
# @Time : 2019/7/23 19:48
# @Author : naihai

"""
激活函数
"""

import numpy as np


class Base(object):
    """
    激活函数基类
    """

    def __call__(self, x):
        pass

    def gradient(self, x):
        pass


class Sigmoid(Base):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1.0 - self.__call__(x))


class SoftMax(Base):
    def __call__(self, x):
        ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)


class TanH(Base):

    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class ReLU(Base):
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU(Base):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class ELU(Base):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha * np.exp(x))


class SELU(Base):
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))


class SoftPlus(Base):
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))
