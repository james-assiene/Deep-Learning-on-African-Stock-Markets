# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:56:00 2017

@author: User
"""

import backtrader as bt
import talib as ta
import numpy as np
import datetime
import math
from keras.models import load_model

from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing

from sklearn import preprocessing

class MyStrategy(bt.SignalStrategy):
    
    def __init__(self):
#        self.model = load_model("jack_cnn_convpool_0_conv_2_fcn_0_filters_64_64_kernel_5_3_fcs__pool_l1_001_adam_pt_100_lr_1em3_50ep_btch32_all_tai.h5")
        self.model = load_model("final_ tun_cnn_convpool_0_conv_2_fcn_0_filters_64_64_kernel_5_5_fcs__pool_do_03_adam_pt_100_lr_1em3_50ep_btch32_all_tai.h5")
        self.DA = DataAccess()
        self.DP = DataPreprocessing()
        self.past_timeframe = 100
        self.stock_name = "googl"
        self.enter_position_probability = 0.9
        self.exit_position_probability = 0.5
        self.order = None
        self.order_size = 0
        self.dataclose = self.datas[0].close
        self.create_tai()
                                   
    def create_tai(self):
        #VOLUMNE_INDICATORS
        self.ta_indicators = []
        self.ta_indicators.append({"name": "AD", "voi": "Close", "params": {}})
        self.ta_indicators.append({"name": "ADOSC", "voi": "Close", "params": {"fastperiod": 3, "slowperiod": 10}})
        self.ta_indicators.append({"name": "OBV", "voi": "Close", "params": {}})
        
        #VOLATILITY_INDICATORS
        self.ta_indicators.append({"name": "ATR", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "TRANGE", "voi": "Close", "params": {}})
        
        #PATTERN_RECOGNITION
        self.ta_indicators.append({"name": "CDLDRAGONFLYDOJI", "voi": "Close", "params": {}})
        
        #OVERLAP_STUDIES
        self.ta_indicators.append({"name": "SMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "EMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "DEMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "KAMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "TEMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "TRIMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "WMA", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "MIDPOINT", "voi": "Close", "params": {"timeperiod": 7}})
        self.ta_indicators.append({"name": "MA", "voi": "Close", "params": {"timeperiod": 3, "matype": ta.MA_Type.KAMA}})
        self.ta_indicators.append({"name": "MAMA", "voi": "Close", "params": {"fastlimit": 0.5, "slowlimit": 0.05}}) #bug
        self.ta_indicators.append({"name": "SAR", "voi": "Close", "params": {"acceleration": 0.02, "maximum": 0.2}}) 
        self.ta_indicators.append({"name": "SAREXT", "voi": "Close", "params": {}}) #there's a lot of optionnal parameters
        self.ta_indicators.append({"name": "T3", "voi": "Close", "params": {"timeperiod": 5, "vfactor": 0.7}}) #bug
        self.ta_indicators.append({"name": "MIDPRICE", "voi": "Close", "params": {"timeperiod": 3}}) #bug
        #self.ta_indicators.append({"name": "MAVP", "voi": "Close", "params": {"minperiod": 2, "maxperiod": 30, "periods": np.array([float(x) for x in [1,2,3]])}}) #input lengths are different
        #self.ta_indicators.append({"name": "HT_TRENDLINE", "voi": "Close", "params": {}})
        self.ta_indicators.append({"name": "BBANDS", "voi": "Close", "params": {"timeperiod": 3, "nbdevup": 3, "nbdevdn": 3, "matype": ta.MA_Type.DEMA}})
        #MOMENTUM
        self.ta_indicators.append({"name": "MOM", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "ROC", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "RSI", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "TRIX", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "WILLR", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "ULTOSC", "voi": "Close", "params": {"timeperiod1": 3, "timeperiod2": 4, "timeperiod3": 5}})
        self.ta_indicators.append({"name": "CMO", "voi": "Close", "params": {"timeperiod": 3}}) #bug
        self.ta_indicators.append({"name": "ADX", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "CCI", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "MINUS_DI", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "PLUS_DI", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "DX", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "AROON", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "AROONOSC", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "MINUS_DM", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "PLUS_DM", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "BOP", "voi": "Close", "params": {}})
        self.ta_indicators.append({"name": "MFI", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "ADXR", "voi": "Close", "params": {"timeperiod": 3}})
        self.ta_indicators.append({"name": "APO", "voi": "Close", "params": {"slowperiod": 7, "fastperiod": 3, "matype": ta.MA_Type.T3}}) #bug
        self.ta_indicators.append({"name": "PPO", "voi": "Close", "params": {"slowperiod": 7, "fastperiod": 3, "matype": ta.MA_Type.T3}}) #bug
        self.ta_indicators.append({"name": "MACD", "voi": "Close", "params": {"slowperiod": 12, "fastperiod": 26, "signalperiod": 9}}) #bug
        self.ta_indicators.append({"name": "MACDEXT", "voi": "Close", "params": {"slowperiod": 12,"fastperiod": 26, "signalperiod": 9, "slowmatype": ta.MA_Type.EMA, "fastmatype": ta.MA_Type.EMA, "signalmatype": ta.MA_Type.EMA}}) #bug
        self.ta_indicators.append({"name": "STOCH", "voi": "Close", "params": {"slowk_period": 12, "fastk_period": 26, "slowd_period": 12, "slowd_matype": ta.MA_Type.EMA, "slowk_period": 12, "slowk_matype": ta.MA_Type.EMA}}) #bug
        self.ta_indicators.append({"name": "STOCHF", "voi": "Close", "params": {"fastd_period": 12, "fastk_period": 12, "fastd_matype": ta.MA_Type.EMA}}) #bug
        self.ta_indicators.append({"name": "STOCHRSI", "voi": "Close", "params": {"fastd_period": 12, "fastk_period": 12, "fastd_matype": ta.MA_Type.EMA, "timeperiod": 3}}) #bug
                
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
                self.buy_price = order.executed.price
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None
        
    def next(self):
        # Simply log the closing price of the series from the reference
        
        if self.order:
            return
        
        end_date = self.datas[0].datetime.date(0)
        start_date = end_date - datetime.timedelta(days=(self.past_timeframe+150))
        
#        data = self.DA.get_stocks(self.stock_name, start=start_date, end=end_date, source="google")
        data = self.DA.get_stock_from_csv("tun index 10y.csv")
        data.dropna(inplace=True)
        data = data.sort()
        X_tai = self.DP.get_tai_dataset(data.ix[start_date:end_date], indicators=self.ta_indicators)
        X = []
        
        df_subset = X_tai.iloc[-self.past_timeframe:]
        X.append(preprocessing.scale(df_subset.values))
        X = np.array(X)
                
        
        probs = self.model.predict_proba(X)
        decrease_probability = probs[0][0]
        increase_probability = probs[0][1]
        
        if not self.position:
            
#            self.order_size = math.floor(self.broker.get_cash() / self.datas[0][0])
            self.order_size = 1
            
            if increase_probability > self.enter_position_probability:
                self.log("BUY CREATED : Close {}".format(self.datas[0][0]))
                self.order = self.buy(size=self.order_size)
                
        else:
            
            if self.dataclose[0] > self.buy_price:
                self.log("SELL CREATED : Close {}".format(self.datas[0][0]))
                self.order = self.sell(size=self.order_size)
        
        self.log('Close, %.2f' % self.dataclose[0])