# -*- coding:utf-8 -*-
# @Time : 2019/7/23 19:48
# @Author : naihai


"""
优化器
"""


class Base(object):
    def update(self, w, grad_wrt_w):
        raise NotImplementedError


class SGD(Base):
    def __init__(self, learning_rate=0.01, momentum=0):
        pass

    def update(self, w, grad_wrt_w):
        pass