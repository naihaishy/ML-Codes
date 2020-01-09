# -*- coding:utf-8 -*-
# @Time : 2019/8/2 9:39
# @Author : naihai

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

from factorization_machine import FM, FFM, FMFast


def fm_binary_classification():
    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    model = FM(1, 0.1, [0.001, 0.001, 0.001], 10)
    model.fit(X_train, y_train, 10)

    pres = model.predict(X_test)
    print(accuracy_score(y_test, pres))


def fm_binary_classification_titanic():
    data = pd.read_csv("E:\Recommend-Projects\\xlearn-master\demo\classification\\titanic\\titanic_train.txt", sep='\t',
                       header=None)
    X = data[data.columns[1:]]
    y = data[0].values
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    model = FM(1, 0.02, [0.001, 0.001, 0.01], 100)
    model.fit(X_train, y_train, 100)

    pres = model.predict(X_test)
    print(accuracy_score(y_test, pres))


def fm_fast_binary_classification_titanic():
    data = pd.read_csv("E:\Recommend-Projects\\xlearn-master\demo\classification\\titanic\\titanic_train.txt", sep='\t',
                       header=None)
    X = data[data.columns[1:]]
    y = data[0].values
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    for lr in [0.001, 0.002, 0.005, 0.01, 0.02]:
        model = FMFast(1, lr, [0.1, 0.1, 0.1], 100)
        model.fit(X_train, y_train, 100)

        pres = model.predict(X_test)
        print(accuracy_score(y_test, pres))


def fm_fast_binary_classification():
    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    model = FMFast(1, 0.1, [0.01, 0.01, 0.01], 10)
    model.fit(X_train, y_train, 100)

    pres = model.predict(X_test)

    print(accuracy_score(y_test, pres))


if __name__ == '__main__':
    fm_fast_binary_classification_titanic()
