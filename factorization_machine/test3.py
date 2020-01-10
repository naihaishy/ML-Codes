# -*- coding:utf-8 -*-
# @Time : 2020/1/9 9:51
# @Author : naihai

import xlearn as xl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

if __name__ == '__main__':
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

    train_data = xl.DMatrix(data=X_train, label=y_train)
    test_data = xl.DMatrix(data=X_test, label=y_test)

    fm = xl.create_fm()
    fm.setTrain(train_data)
    fm.setTest(test_data)

    param = {
        'task': 'reg',
        'lr': 0.1,
        'lambda': 0.02,
        'k': 100,
        'epoch': 100,
        'metric': 'rmse'}
    fm.fit(param, "./model.out")

    pres = fm.predict("./model.out")

    print(mean_squared_error(y_test, pres))
