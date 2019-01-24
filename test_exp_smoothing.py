# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:07:20 2017

@author: User
"""



import numpy as np


from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing


DA = DataAccess()
DP = DataPreprocessing()

value_of_interest = "Close"

#data = DA.get_stocks("jse:jse", "2007-1-1", "2017-1-1", source="google")
#data.dropna(inplace=True)
stock_name = "egx30"
data = DA.get_stock_from_csv(stock_name + " index 10y.csv")

past_timeframe=100
X,Y = DP.create_dataset_without_tai(data, past_timeframe=past_timeframe, preprocess=False)

X_train, Y_train, X_test, Y_test = DP.create_train_test_datasets(X,Y, train_percentage=0.8, with_tai=False)


def get_exp_smoothing(values, alpha = 0.5):    

    exp_smoothing = []
    exp_smoothing.append(values[0])
    for i in range(1, len(values)):
        exp_smoothing.append(alpha * values[i] + (1 - alpha) * exp_smoothing[i - 1])
        
    return exp_smoothing
        
        
#values = data["Close"][0:len(X_train)]

def get_exp_smoothing_predictions(dataset, alpha = 0.5):
    pred = []
    for i in range(dataset.shape[0]):
        close_index = 3
        values = []
        for j in range(len(dataset[i])):
            values.append(dataset[i][j][close_index])
        exp_smoothing = get_exp_smoothing(values, alpha)
        pred.append(1 if exp_smoothing[-1] > values[-1] else 0)
        
    return pred

def exp_smoothing_accuracy(predictions, labels):
    return sum(predictions == labels["Rising"])/len(predictions)

def train_exp_smoothing(dataset_train, dataset_test):
    scores = []
    alpha_range = np.arange (0, 1, 0.01)
    for alpha in alpha_range:
        pred = get_exp_smoothing_predictions(dataset_train, alpha)
        scores.append(exp_smoothing_accuracy(pred, dataset_test))
    
    scores = np.array(scores)
    print(scores)
    print("best score : {} - alpha : {}".format(np.amax(scores), alpha_range[np.argmax(scores)]))
    
    return alpha_range[np.argmax(scores)]

best_alpha = train_exp_smoothing(X_train, Y_train)
pred = get_exp_smoothing_predictions(X_test, best_alpha)
accuracy = exp_smoothing_accuracy(pred, Y_test)

print("test : {}".format(accuracy))