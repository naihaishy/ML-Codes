# -*- coding:utf-8 -*-
# @Time : 2020/1/12 15:56
# @Author : naihai

"""
Python 调用 C++ Api 底层实现的Factorization Machine
"""

import numpy as np
import pandas as pd
import ctypes


def load_lib():
    return ctypes.cdll.LoadLibrary("E:\CLion\Projects\FM\cmake-build-debug\libFM.dll")


_LIB = load_lib()


class DMatrix(object):
    def __init__(self, data, label=None):
        """
        :param data: Numpy 2D or Pandas DataFrame
        :param label: Python List or Numpy 1D or Pandas DataFrame/Series
        """
        self.__handle = ctypes.c_void_p()  # 指向DMatrix对象的内存区域
        if isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame):
            self._init_from_npy_or_pd(data, label)
        else:
            raise ValueError('Input data must be numpy.ndarray or pandas.DataFrame')

    def _init_from_npy_or_pd(self, data, label=None):
        """
        从numpy or pandas 数据结构构建对象
        :param data: numpy.ndarray, pandas.DataFrame
        :param label: list, numpy.ndarray, pandas.DataFrame, pandas.Series
        :return:
        """
        if len(data.shape) != 2:
            raise ValueError('Input numpy.ndarray or pandas.DataFrame must be 2 dimensional')

        # 统一使用numpy 2d array 操作
        if isinstance(data, pd.DataFrame):
            data = data.values
        # 将二维结构转换为numpy一维数组 内存连续存储
        data = np.array(data.reshape(data.size), copy=False, dtype=np.float32)
        # 获取底层内存存储的指针 指向c-types类型对象
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        label_ptr = None

        if label is not None:
            # 将label转换为numpy array 1d
            if isinstance(label, pd.DataFrame):
                label = label.values
            if isinstance(label, pd.Series):
                label = label.values
            if isinstance(label, list):
                label = np.array(label)
            if isinstance(label, np.ndarray):
                if len(label.shape) > 2:
                    raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional '
                                     'with one dimensional is 1, current label shape is {0}'.format(label.shape))
                if (len(label.shape) == 2) and (label.shape[0] != 1) and (label.shape[1] != 1):
                    print(len(label.shape))
                    raise ValueError('Input numpy.ndarray of label must be 1 dimensional or 2 dimensional '
                                     'with one dimensional is 1, current label shape is {0}'.format(label.shape))
                if label.size != data.shape[0]:
                    raise ValueError('Input label must has same elements as the data rows, {0} != {1} '.format(
                        label.size, data.shape[0]))
                # 将二维结构转换为numpy一维数组
                label = np.array(label.reshape(label.size), copy=False, dtype=np.float32)
                label_ptr = label.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            else:
                raise ValueError("Input label must be list or numpy.ndarray or pandas.DataFrame/Series")
        _LIB.FMCreateDataFromMat(data_ptr,
                                 ctypes.c_uint64(data.shape[0]),
                                 ctypes.c_uint64(data.shape[1]),
                                 label_ptr,
                                 ctypes.byref(self.__handle))

    @property
    def handle(self):
        return self.__handle

    def __del__(self):
        """
        release the resource of DMatrix
        """
        _LIB.FMDataFree(ctypes.byref(self.__handle))
        self.__handle = None
