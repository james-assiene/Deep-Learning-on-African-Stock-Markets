#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:05:16 2017

@author: squall
"""

import pandas as pd
import talib as ta
import numpy as np
import copy
import multiprocessing
import logging

from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing

from TAIndicatorSelection import TAIndicatorSelection

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from multiprocessing import Pool, freeze_support
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy


DA = DataAccess()
DP = DataPreprocessing()

print("pingouin++")

value_of_interest = "Close"
ta_indicators = []

#VOLUMNE_INDICATORS
ta_indicators.append({"name": "AD", "voi": "Close", "params": {}})
ta_indicators.append({"name": "ADOSC", "voi": "Close", "params": {"fastperiod": 3, "slowperiod": 10}})
ta_indicators.append({"name": "OBV", "voi": "Close", "params": {}})

#VOLATILITY_INDICATORS
ta_indicators.append({"name": "ATR", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "TRANGE", "voi": "Close", "params": {}})

#PATTERN_RECOGNITION
ta_indicators.append({"name": "CDLDRAGONFLYDOJI", "voi": "Close", "params": {}})

#OVERLAP_STUDIES
ta_indicators.append({"name": "SMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "EMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "DEMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "KAMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "TEMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "TRIMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "WMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "MIDPOINT", "voi": "Close", "params": {"timeperiod": 7}})
ta_indicators.append({"name": "MA", "voi": "Close", "params": {"timeperiod": 3, "matype": ta.MA_Type.KAMA}})
ta_indicators.append({"name": "MAMA", "voi": "Close", "params": {"fastlimit": 0.5, "slowlimit": 0.05}}) #bug
ta_indicators.append({"name": "SAR", "voi": "Close", "params": {"acceleration": 0.02, "maximum": 0.2}}) 
ta_indicators.append({"name": "SAREXT", "voi": "Close", "params": {}}) #there's a lot of optionnal parameters
ta_indicators.append({"name": "T3", "voi": "Close", "params": {"timeperiod": 5, "vfactor": 0.7}}) #bug
ta_indicators.append({"name": "MIDPRICE", "voi": "Close", "params": {"timeperiod": 3}}) #bug
#ta_indicators.append({"name": "MAVP", "voi": "Close", "params": {"minperiod": 2, "maxperiod": 30, "periods": np.array([float(x) for x in [1,2,3]])}}) #input lengths are different
#ta_indicators.append({"name": "HT_TRENDLINE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "BBANDS", "voi": "Close", "params": {"timeperiod": 3, "nbdevup": 3, "nbdevdn": 3, "matype": ta.MA_Type.DEMA}})
#MOMENTUM
ta_indicators.append({"name": "MOM", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "ROC", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "RSI", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "TRIX", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "WILLR", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "ULTOSC", "voi": "Close", "params": {"timeperiod1": 3, "timeperiod2": 4, "timeperiod3": 5}})
ta_indicators.append({"name": "CMO", "voi": "Close", "params": {"timeperiod": 3}}) #bug
ta_indicators.append({"name": "ADX", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "CCI", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "MINUS_DI", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "PLUS_DI", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "DX", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "AROON", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "AROONOSC", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "MINUS_DM", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "PLUS_DM", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "BOP", "voi": "Close", "params": {}})
ta_indicators.append({"name": "MFI", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "ADXR", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "APO", "voi": "Close", "params": {"slowperiod": 7, "fastperiod": 3, "matype": ta.MA_Type.T3}}) #bug
ta_indicators.append({"name": "PPO", "voi": "Close", "params": {"slowperiod": 7, "fastperiod": 3, "matype": ta.MA_Type.T3}}) #bug
ta_indicators.append({"name": "MACD", "voi": "Close", "params": {"slowperiod": 12, "fastperiod": 26, "signalperiod": 9}}) #bug
ta_indicators.append({"name": "MACDEXT", "voi": "Close", "params": {"slowperiod": 12,"fastperiod": 26, "signalperiod": 9, "slowmatype": ta.MA_Type.EMA, "fastmatype": ta.MA_Type.EMA, "signalmatype": ta.MA_Type.EMA}}) #bug
ta_indicators.append({"name": "STOCH", "voi": "Close", "params": {"slowk_period": 12, "fastk_period": 26, "slowd_period": 12, "slowd_matype": ta.MA_Type.EMA, "slowk_period": 12, "slowk_matype": ta.MA_Type.EMA}}) #bug
ta_indicators.append({"name": "STOCHF", "voi": "Close", "params": {"fastd_period": 12, "fastk_period": 12, "fastd_matype": ta.MA_Type.EMA}}) #bug
ta_indicators.append({"name": "STOCHRSI", "voi": "Close", "params": {"fastd_period": 12, "fastk_period": 12, "fastd_matype": ta.MA_Type.EMA, "timeperiod": 3}}) #bug

all_ta_indicators = copy.deepcopy(ta_indicators)

ta_indicators = []
ta_indicators.append({"name": "SMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "MACD", "voi": "Close", "params": {"slowperiod": 12, "fastperiod": 26, "signalperiod": 9}}) #bug

stock_name = "qqq"
data = DA.get_stocks(stock_name, "2011-1-1", "2017-5-12")

def find_best_technical_indicators():
    ta_selector = TAIndicatorSelection()

    from timeit import default_timer as timer
    
    
    print("reinheart")
    
    
    if __name__ == "__main__":
        print("pingouin")
        freeze_support()
        print("inside the main")
    #    multiprocessing.log_to_stderr()
    #    logger = multiprocessing.get_logger()
    #    logger.setLevel(logging.DEBUG)
    #    start = timer()
    #    final_ta, best_ta, res = ta_selector.greedy_forward_sp(data)
    #    end= timer()
    #    print("Single process {}".format(end - start))
    #    start = timer()
    #    final_ta, best_ta, res = ta_selector.greedy_forward_mt(data)
    #    end= timer()
    #    print("Multi threaded {}".format(end - start))
        start = timer()
        final_ta, best_ta, res = ta_selector.greedy_forward_mp(data)
        end= timer()
        print("Multi process {}".format(end - start))


#future = DA.get_stocks("aapl", "2013-1-30", "2013-1-31")

#inputs = DP.get_technical_indicators(data, ta_subset = ta_indicators)
X,Y = DP.create_dataset(data, past_timeframe=100, indicators=ta_indicators)

X_train, Y_train, X_test, Y_test = DP.create_train_test_datasets(X,Y, train_percentage=0.99)

cols = X_train.columns
num_samples = len(X_train)
num_features = len(X_train.columns)

xa = 3

#def cross_validation_evaluation(models):
#    results = []
#
#    for modelInformations in models:
#        X_train_matrix, Y_train_matrix, X_test_matrix, Y_test_matrix = DP.datasets_as_matrices(X_train, Y_train, X_test, Y_test, datasets_form=modelInformations["datasets_form"])
#    
#        model2 = KerasClassifier(build_fn=modelInformations["model"], epochs=150, batch_size=10, verbose=0)
#        # evaluate using 10-fold cross validation
#        kfold = StratifiedKFold(n_splits=5, shuffle=True)
#        result = cross_val_score(model2, X_train_matrix, Y_train_matrix.reshape(num_samples,), cv=kfold)
#        results.append({"name": modelInformations["name"], "values": result})
#        print(result.mean())
#        
#    return results
#
num_samples = len(X_train)
num_features = len(X_train.columns)

from keras.wrappers.scikit_learn import KerasClassifier
from MLPModel import MLPModel
from LSTMModel import LSTMModel
from SimpleRNNModel import SimpleRNNModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.np_utils import to_categorical

def grid_search(model_informations):
    model = KerasClassifier(build_fn=model_informations["model"])

    optimizers = ['adam']
    init = ['glorot_normal']
    epochs = [100]
    batches = [50]
    neurons = range(100, 1600, 100)
    
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    X_train_matrix, Y_train_matrix, X_test_matrix, Y_test_matrix = DP.datasets_as_matrices(X_train, Y_train, X_test, Y_test, datasets_form=model_informations["datasets_form"])
    grid_result = grid.fit(X_train_matrix, Y_train_matrix)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
    	print("%f (%f) with: %r" % (mean, stdev, param))
        
    return means, stds, params

def plot_grid_search_results(models_results, security_name):
    import matplotlib.pyplot as plt

    xvalues = range(100, 1600, 100)
    folds = np.array(xvalues)
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    
    for current_model in models_results:
        ax1.plot(folds, current_model["values"], label=current_model["name"], color=current_model["color"], marker = "o")
        
    plt.xticks(folds)
    plt.xlabel("Neurons")
    handles, labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
    ax1.grid('on')
    
    plt.legend(loc='best')
    plt.savefig(security_name + ".png")
    
    plt.show()


models = []
models_results = []

models.append({"name": "MLP", "model": MLPModel(num_samples=num_samples, num_features=num_features), "datasets_form": "Normal", "color": "c"})
models.append({"name": "LSTM", "model": LSTMModel(num_samples=num_samples, num_features=num_features), "datasets_form": "RNN", "color" : "g"})
models.append({"name": "Simple RNN", "model": SimpleRNNModel(num_samples=num_samples, num_features=num_features), "datasets_form": "RNN", "color": "r"})

for model in models:
    means, stds, params = grid_search(model)
    models_results.append({"name": model["name"], "values": means, "color": model["color"]})
    
#plot_grid_search_results(models_results, stock_name)

#model = ANNs.create_mlp_model(num_features)



#models = []
#models.append({"name": "MLP", "model": MLPModel(num_samples=num_samples, num_features=num_features), "datasets_form": "Normal"})
#models.append({"name": "LSTM", "model": LSTMModel(num_samples=num_samples, num_features=num_features), "datasets_form": "RNN"})
#models.append({"name": "Simple RNN", "model": SimpleRNNModel(num_samples=num_samples, num_features=num_features), "datasets_form": "RNN"})
#models_results = cross_validation_evaluation(models=models) 

#model.fit(X_train_matrix, to_categorical(Y_train_matrix), epochs=5, batch_size=32)
#loss_and_metrics = model.evaluate(X_test_matrix, to_categorical(Y_test_matrix), batch_size=128)
    

#print(loss_and_metrics)