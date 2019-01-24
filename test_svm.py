# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:09:55 2017

@author: User
"""


import numpy as np


from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing


DA = DataAccess()
DP = DataPreprocessing()

value_of_interest = "Close"

data = DA.get_stocks("jse:jse", "2007-1-1", "2017-1-1", source="google")
data.dropna(inplace=True)
#stock_name = "egx30"
#data = DA.get_stock_from_csv(stock_name + " index 10y.csv")

past_timeframe=100
X,Y = DP.create_dataset_without_tai(data, past_timeframe=past_timeframe)

X_train, Y_train, X_test, Y_test = DP.create_train_test_datasets(X,Y, train_percentage=0.8, with_tai=False)

from sklearn.svm import SVC
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
clf = SVC()
clf.fit(X_train,  np.reshape(Y_train.values, Y_train.shape[0]))
res = clf.score(X_test, np.reshape(Y_test.values, Y_test.shape[0]))

print("score : {}".format(res))
