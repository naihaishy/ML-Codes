# -*- coding:utf-8 -*-
# @Time : 2019/8/2 9:39
# @Author : naihai

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

from factorization_machine import FM, FFM


def fm_binary_classification():
    """
    :return:
    """
    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    model = FM(1, 0.1, 0.1, 100)
    model.fit(X_train, y_train, 100)

    pres = model.predict(X_test)

    print(accuracy_score(y_test, pres))


def fm_regression():
    data = pd.read_csv("E:\Scala\projects\Recommend\data\house_price_train.txt", sep='\t', header=None)
    data = data
    # get train X, y
    X_train = data[data.columns[1:]]
    y_train = data[0]
    # 预处理
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    print(X_train.shape)

    model = FM(0, 0.01, 0.1, 100)
    model.fit(X_train, y_train, 1000)


def ffm_binary_classification():
    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    feature_dict = dict()

    for i in range(64):
        feature_dict[i] = i % 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

    model = FFM(0.1, 20, 10, feature_dict)
    model.fit(X_train, y_train, 10)

    pres = model.predict(X_test)

    print(accuracy_score(y_test, pres))


if __name__ == '__main__':
    fm_regression()
