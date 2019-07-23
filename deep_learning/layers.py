# -*- coding:utf-8 -*-
# @Time : 2019/7/23 19:48
# @Author : naihai

"""
神经网络层
"""


class Layer(object):
    def __init__(self):
        self.input_shape = None
        pass

    def __call__(self, *args, **kwargs):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
