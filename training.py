# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 00:40:52 2017

@author: User
"""

import talib as ta
import pandas as pd
import numpy as np
import copy
import keras
from copy import deepcopy

from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing

from MLPModel import MLPModel
from LSTMModel import LSTMModel
from SimpleRNNModel import SimpleRNNModel
from ConvnetModel import ConvNetModel, ConvNetModel2L, ConvNetModelNL
from LSTMNoTai import LSTMNoTaiModel
from VanillaRNNNoTai import VanillaRNNNoTaiModel


DA = DataAccess()
DP = DataPreprocessing()

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

def dataset_from_multiple_stocks(stock_list, past_timeframe = 100):
    
    stock_name = "spy"
    X_tests = []
    Y_tests = []
    X_train = []
    Y_train = []
    
    for idx, stock_name in enumerate(stock_list):
    
        data = DA.get_stocks(stock_name, "2000-1-1", "2017-7-07", source="google")
        data.dropna(inplace=True)
        
        X_tai = DP.get_tai_dataset(data, indicators=all_ta_indicators)
        X,Y = DP.create_dataset_without_tai(X_tai, past_timeframe=past_timeframe)
        
        X_train_i, Y_train_i, X_test_i, Y_test_i = DP.create_train_test_datasets(X,Y, train_percentage=0.9, with_tai=False)
        
        if idx == 0:
            X_train = X_train_i
            Y_train = Y_train_i
            
        else:
            X_train = np.concatenate((X_train, X_train_i))
            Y_train = pd.concat([Y_train, Y_train_i])
            
        X_tests.append(X_test_i)
        Y_tests.append(Y_test_i)
        
    return X_train, Y_train, X_tests, Y_tests

ta_indicators = []
ta_indicators.append({"name": "SMA", "voi": "Close", "params": {"timeperiod": 3}})
ta_indicators.append({"name": "MACD", "voi": "Close", "params": {"slowperiod": 12, "fastperiod": 26, "signalperiod": 9}}) #bug

stocks = ["spy", "aapl", "msft", "nvda"]
past_timeframe = 100
X_train, Y_train, X_tests, Y_tests = dataset_from_multiple_stocks(stocks, past_timeframe=past_timeframe)

num_samples = len(X_train)
num_features = len(X_train[0][0])

possible_filters = [16, 32, 64, 128]
possible_kernel_size = [3, 5, 7]
possible_conv_num = [1, 2, 3]
possible_conv_pool_num = [1, 2, 3]
possible_fc_num = [1, 2, 3]
possible_fc_size = [32, 64, 128]

filters = [64]
kernel_size = [7]
pool_size = [2]
fc_size = []
conv_num = len(filters)
conv_pool_num = len(pool_size)
fc_num = len(fc_size)

model_name = "jack_cnn_" + "convpool_" + str(conv_pool_num) + "_conv_" + str(conv_num) + "_fcn_" + str(fc_num) + "_"
model_name+= "filters_" + "_".join(str(x) for x in filters) + "_kernel_" + "_".join(str(x) for x in kernel_size) + "_"
model_name+= "fcs_" + "_".join(str(x) for x in fc_size) + "_pool_" + "_".join(str(x) for x in pool_size)
        

convnet_model = ConvNetModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
convnet2l_model = ConvNetModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
convnetnl_model = ConvNetModelNL(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
lstm_model = LSTMNoTaiModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
vanilla_rnn_model = VanillaRNNNoTaiModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
#conv_num, conv_pool_num, fc_num, filters, kernel_size, fc_size, pool_size
model = convnetnl_model(conv_num, conv_pool_num, fc_num, filters, kernel_size, fc_size, pool_size)

#l1_001_
model_name+= "l2_001_adam_pt_100_lr_1em3_60ep_btch64_all_tai"

log_dir = "_".join(stocks) + "/" + model_name

tb_callback = keras.callbacks.TensorBoard(histogram_freq=5, write_grads=True, log_dir=log_dir)
model.fit(X_train, Y_train.values, batch_size=64, epochs=60, callbacks=[tb_callback], validation_data=(X_tests[0], Y_tests[0].values))

for idx, stock in enumerate(stocks):
    score = model.evaluate(X_tests[idx], Y_tests[idx].values, batch_size=16)
    print("score on {} : {}".format(stock, score))
    
model.save(model_name + ".h5")