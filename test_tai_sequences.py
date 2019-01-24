# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:05:12 2017

@author: User
"""

import talib as ta
import pandas as pd
import copy
import keras
from copy import deepcopy

from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing

from MLPModel import MLPModel
from LSTMModel import LSTMModel
from SimpleRNNModel import SimpleRNNModel
from ConvnetModel import ConvNetModel, ConvNetModelNL
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
ta_indicators.append({"name": "CDL2CROWS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDL3BLACKCROWS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDL3INSIDE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDL3LINESTRIKE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDL3OUTSIDE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDL3STARSINSOUTH", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDL3WHITESOLDIERS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLABANDONEDBABY", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLADVANCEBLOCK", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLBELTHOLD", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLBREAKAWAY", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLCLOSINGMARUBOZU", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLCONCEALBABYSWALL", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLCOUNTERATTACK", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLDARKCLOUDCOVER", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLDOJI", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLDOJISTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLENGULFING", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLEVENINGDOJISTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLEVENINGSTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLGAPSIDESIDEWHITE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLGRAVESTONEDOJI", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHAMMER", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHANGINGMAN", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHARAMI", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHARAMICROSS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHIGHWAVE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHIKKAKE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHIKKAKEMOD", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLHOMINGPIGEON", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLIDENTICAL3CROWS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLINNECK", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLINVERTEDHAMMER", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLKICKING", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLKICKINGBYLENGTH", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLLADDERBOTTOM", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLLONGLEGGEDDOJI", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLLONGLINE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLMARUBOZU", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLMATCHINGLOW", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLMATHOLD", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLMORNINGDOJISTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLMORNINGSTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLONNECK", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLPIERCING", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLRICKSHAWMAN", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLRISEFALL3METHODS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLSEPARATINGLINES", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLSHOOTINGSTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLSHORTLINE", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLSPINNINGTOP", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLSTALLEDPATTERN", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLSTICKSANDWICH", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLTAKURI", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLTASUKIGAP", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLTHRUSTING", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLTRISTAR", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLUNIQUE3RIVER", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLUPSIDEGAP2CROWS", "voi": "Close", "params": {}})
ta_indicators.append({"name": "CDLXSIDEGAP3METHODS", "voi": "Close", "params": {}})

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

stock_name = "tun"
#data = DA.get_stocks("jse:jse", "2007-1-1", "2017-1-1", source="google")
#data.dropna(inplace=True)
#egx = pd.read_csv("nikkei_stock_average_daily_en.csv", decimal=".", sep=",", quotechar='"')
#egx = egx[(egx != 0).all(1)]
#egx = egx.set_index(pd.DatetimeIndex(egx["Date of Data"]))
#egx.drop(["Date of Data"], axis=1, inplace=True)

stock_name = "ngse"
data = DA.get_stock_from_csv(stock_name + " index 10y.csv")

X_tai = DP.get_tai_dataset(data, indicators=all_ta_indicators)
past_timeframe = 100
X,Y = DP.create_dataset_without_tai(X_tai, past_timeframe=past_timeframe)

X_train, Y_train, X_test, Y_test = DP.create_train_test_datasets(X,Y, train_percentage=0.8, with_tai=False)

num_samples = len(X_train)
num_features = len(X_train[0][0])

possible_filters = [16, 32, 64, 128]
possible_kernel_size = [3, 5, 7]
possible_conv_num = [1, 2, 3]
possible_conv_pool_num = [1, 2, 3]
possible_fc_num = [1, 2, 3]
possible_fc_size = [32, 64, 128]

filters = [64,64]
kernel_size = [5,5]
pool_size = []
fc_size = []
conv_num = len(filters)
conv_pool_num = len(pool_size)
fc_num = len(fc_size)

model_name = "final_ " + stock_name + "_cnn_" + "convpool_" + str(conv_pool_num) + "_conv_" + str(conv_num) + "_fcn_" + str(fc_num) + "_"
model_name+= "filters_" + "_".join(str(x) for x in filters) + "_kernel_" + "_".join(str(x) for x in kernel_size) + "_"
model_name+= "fcs_" + "_".join(str(x) for x in fc_size) + "_pool_" + "_".join(str(x) for x in pool_size)
        

convnet_model = ConvNetModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
convnet2l_model = ConvNetModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
convnetnl_model = ConvNetModelNL(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
lstm_model = LSTMNoTaiModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
vanilla_rnn_model = VanillaRNNNoTaiModel(num_samples=num_samples, num_features=num_features, sequence_length=past_timeframe)
#conv_num, conv_pool_num, fc_num, filters, kernel_size, fc_size, pool_size
model = convnetnl_model(conv_num, conv_pool_num, fc_num, filters, kernel_size, fc_size, pool_size)

model_name+= "do_03_adam_pt_100_lr_1em3_50ep_btch32_all_tai"

log_dir = stock_name + "/" + model_name


tb_callback = keras.callbacks.TensorBoard(histogram_freq=5, write_grads=True, log_dir=log_dir)
model.fit(X_train, Y_train.values, batch_size=32, epochs=30, callbacks=[tb_callback], validation_data=(X_test, Y_test.values))
score = model.evaluate(X_test, Y_test.values, batch_size=32)

print("score on {} : {}".format(stock_name, score))
    
model.save(model_name + ".h5")