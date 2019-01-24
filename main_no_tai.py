#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:04:22 2017

@author: squall
"""

import pandas as pd
import talib as ta
import numpy as np
import keras

from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing
from MLPModel import MLPModel
from LSTMModel import LSTMModel
from SimpleRNNModel import SimpleRNNModel
from ConvnetModel import ConvNetModel
from LSTMNoTai import LSTMNoTaiModel
from VanillaRNNNoTai import VanillaRNNNoTaiModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.np_utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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


#data = DA.get_stocks("NIKKEI225", "2007-1-1", "2017-5-12", source="fred")
#data.dropna(inplace=True)
#egx = pd.read_csv("egx30 index 10y.csv", decimal=",", sep=";")
#egx = egx[(egx != 0).all(1)]
#egx = egx.set_index(pd.DatetimeIndex(egx["Date"]))
#egx.drop(["Date", "Company"], axis=1, inplace=True)

data = DA.get_stocks("jse:jse", "2007-1-1", "2017-1-1", source="google")
data.dropna(inplace=True)
#stock_name = "nse20"
#data = DA.get_stock_from_csv(stock_name + " index 10y.csv")
#future = DA.get_stocks("aapl", "2013-1-30", "2013-1-31")

#inputs = DP.get_technical_indicators(data, ta_subset = ta_indicators)
past_timeframe=100
X,Y = DP.create_dataset_without_tai(data, past_timeframe=past_timeframe, preprocess=False)

X_train, Y_train, X_test, Y_test = DP.create_train_test_datasets(X,Y, train_percentage=0.8, with_tai=False)

#num_samples = len(X_train)
#num_features = len(X_train[0][0])
#
#convnet_model = ConvNetModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
#lstm_model = LSTMNoTaiModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
#vanilla_rnn_model = VanillaRNNNoTaiModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
#model = convnet_model()
#
#tb_callback = keras.callbacks.TensorBoard(histogram_freq=5, write_grads=True, log_dir="nikkei/cnn_adam_pt_100_lr_1em3_30ep_notai")
#model.fit(X_train, Y_train.values, batch_size=16, epochs=30, callbacks=[tb_callback], validation_data=(X_test, Y_test.values))
#score = model.evaluate(X_test, Y_test.values, batch_size=16)




