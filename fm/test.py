# -*- coding:utf-8 -*-
# @Time : 2019/8/2 9:39
# @Author : naihai

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale, normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fm import FM

if __name__ == '__main__':
    X, y = load_digits(n_class=2, return_X_y=True)
    y[y == 0] = -1
    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

    model = FM(0.1, 0.1, 10)
    model.fit(X_train, y_train, 10)

    pres = model.predict(X_test)

    print(accuracy_score(y_test, pres))
