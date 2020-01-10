# -*- coding:utf-8 -*-
# @Time : 2019/8/2 9:39
# @Author : naihai

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from factorization_machine import FM


def fm_regression():
    data = pd.read_csv("E:\Scala\projects\Recommend\data\house_price_train.txt", sep='\t', header=None)
    X = data[data.columns[1:]]
    y = data[0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    print(X_train.shape)

    # 预处理
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = FM(0, 0.01, [0.1, 0.1, 0.1], 100)
    model.fit(X_train, y_train, 100)

    pres = model.predict(X_test)

    print(mean_squared_error(y_test, pres))


if __name__ == '__main__':
    fm_regression()
