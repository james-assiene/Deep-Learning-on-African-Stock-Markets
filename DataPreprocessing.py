#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:55 2017

@author: squall
"""

import pandas_datareader.data as web
import copy
import pandas as pd
import numpy as np
import urllib.request as Urllib
import datetime as dt
import matplotlib.pyplot as plt
import talib as ta

from sklearn import preprocessing

class DataPreprocessing:
    
    def __init__(self):
        print("")
        
    def get_technical_indicators(self, stock_info, ta_subset = "all"):
        
        result = {}
        
        for function in ta_subset:
            f_name = function["name"]
            f_params = copy.deepcopy(function["params"])
            f_params_concatenated = ""
            for key, value in f_params.items():
                f_params_concatenated+= "_" + key + str(value)
            
            #real
            if f_name in ["SMA","EMA", "MOM", "ROC", "RSI", "TRIX", "DEMA", "HT_TRENDLINE", "KAMA", "MA", "MAVP", "MIDPOINT", "T3", "TEMA", "TRIMA", "WMA"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                ta_name = f_name + f_params_concatenated
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            #high, low, close
            elif f_name in ["ADX", "ADXR", "CCI", "DX", "MINUS_DI", "PLUS_DI", "ULTOSC", "WILLR", "ATR", "TRANGE"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                f_data["close"] = stock_info["Close"].values
                ta_name = f_name + f_params_concatenated
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            #STOCH, 2 outputs
            elif f_name in ["STOCH"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                f_data["close"] = stock_info["Close"].values
                      
                ta_name = f_name + "_SLOWK" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_SLOWD" + f_params_concatenated
                result[ta_name] = tmp[1]
                
            #STOCHF, 2 outputs 
            elif f_name in ["STOCHF"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                f_data["close"] = stock_info["Close"].values
                      
                ta_name = f_name + "_FASTK" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_FASTD" + f_params_concatenated
                result[ta_name] = tmp[1]
                
            #STOCHRSI, 2 outputs 
            elif f_name in ["STOCHF"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                      
                ta_name = f_name + "_FASTK" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_FASTD" + f_params_concatenated
                result[ta_name] = tmp[1]
                
            #MAMA, 2 outputs 
            elif f_name in ["MAMA"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                      
                ta_name = f_name + "_MAMA" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_FAMA" + f_params_concatenated
                result[ta_name] = tmp[1]
                
            
            #high, low, close, open    
            elif f_name in ["BOP", "CDLDRAGONFLYDOJI"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                f_data["close"] = stock_info["Close"].values
                f_data["open"] = stock_info["Open"].values
                ta_name = f_name
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            #high, low, close, volume
            elif f_name in ["MFI", "AD", "ADOSC"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                f_data["close"] = stock_info["Close"].values
                f_data["volume"] = np.array([float(x) for x in stock_info["Volume"].values])
                ta_name = f_name + f_params_concatenated
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            #real, volume
            elif f_name in ["OBV"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                f_data["volume"] = np.array([float(x) for x in stock_info["Volume"].values])
                ta_name = f_name + f_params_concatenated
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            #AROON, 2 outputs
            elif f_name in ["AROON"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                ta_name = f_name + "_DOWN" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_UP" + f_params_concatenated
                result[ta_name] = tmp[1]
                
            #high, low
            elif f_name in ["AROONOSC", "MINUS_DM", "PLUS_DM", "SAR", "SAREXT"]:
                f_voi = function["voi"]
                f_data = f_params
                f_data["high"] = stock_info["High"].values
                f_data["low"] = stock_info["Low"].values
                ta_name = f_name + f_params_concatenated
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            #real, fastperiod, slowperiod, matype
            elif f_name in ["APO", "PPO"]:
                #bug
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                ta_name = f_name + f_params_concatenated
                result[ta_name] = getattr(ta, f_name)(**f_data)
                
            
            #MACD, 3 outputs
            elif f_name in ["MACD", "MACDEXT"]:
                #bug
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                ta_name = f_name + "_MACD" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_MACDSIGNAL" + f_params_concatenated
                result[ta_name] = tmp[1]
                ta_name = f_name + "_MACDHIST" + f_params_concatenated
                result[ta_name] = tmp[2]
                
            #BBANDS, 3 outputs
            elif f_name in ["BBANDS"]:
                #bug
                f_voi = function["voi"]
                f_data = f_params
                f_data["real"] = stock_info[f_voi].values
                ta_name = f_name + "_UPPERBAND" + f_params_concatenated
                tmp = getattr(ta, f_name)(**f_data)
                result[ta_name] = tmp[0]
                ta_name = f_name + "_MIDDLEBAND" + f_params_concatenated
                result[ta_name] = tmp[1]
                ta_name = f_name + "_LOWERBAND" + f_params_concatenated
                result[ta_name] = tmp[2]
            
                
        return result
    
    def create_dataset(self, entire_dataframe, indicators, past_timeframe = 20, stock_price="Close"):
        start_index = 0
        end_index = past_timeframe
        entire_dataframe_rows_number = len(entire_dataframe)
        rising_column_name = "Rising"
        X = pd.DataFrame()
        Y = pd.DataFrame(columns=[rising_column_name])
        tmp_row = {}
        
        while end_index < entire_dataframe_rows_number:
            df_subset = entire_dataframe.iloc[start_index:end_index]
            row = self.get_technical_indicators(df_subset, ta_subset=indicators)
            for indicator, indicator_values in row.items():
                for idx, indicator_value in enumerate(indicator_values):
                    tmp_row[indicator + "_" + str(idx)] = [indicator_value]
                    
            print(str(start_index) + " " + str(entire_dataframe_rows_number))
            X_row = pd.DataFrame(tmp_row)
            X = X.append(X_row)
            is_price_rising =  1 if (entire_dataframe[stock_price].iloc[end_index] > entire_dataframe[stock_price].iloc[end_index - 1]) else 0
            Y = Y.append([{rising_column_name: is_price_rising}])
            start_index+= 1
            end_index+= 1
            tmp_row = {}
            
        X.dropna(axis=1, inplace=True)
        return X,Y
    
    def create_dataset_with_tai_sequences(self, entire_dataframe, indicators, past_timeframe = 20, stock_price="Close"):
        start_index = 0
        end_index = past_timeframe
        entire_dataframe_rows_number = len(entire_dataframe)
        rising_column_name = "Rising"
        X = pd.DataFrame()
        Y = pd.DataFrame(columns=[rising_column_name])
        tmp_row = {}
        
        while end_index < entire_dataframe_rows_number:
            df_subset = entire_dataframe.iloc[start_index:end_index]
            row = self.get_technical_indicators(df_subset, ta_subset=indicators)
            for indicator, indicator_values in row.items():
                for idx, indicator_value in enumerate(indicator_values):
                    tmp_row[indicator + "_" + str(idx)] = [indicator_value]
            X_row = pd.DataFrame(tmp_row)
            X = X.append(X_row)
            is_price_rising =  1 if (entire_dataframe[stock_price].iloc[end_index] > entire_dataframe[stock_price].iloc[end_index - 1]) else 0
            Y = Y.append([{rising_column_name: is_price_rising}])
            start_index+= 1
            end_index+= 1
            tmp_row = {}
            
        X.dropna(axis=1, inplace=True)
        return X,Y
    
    def get_tai_dataset(self, entire_dataframe, indicators):
        
        start_index = 0
        entire_dataframe_rows_number = len(entire_dataframe)
        rising_column_name = "Rising"
        X = pd.DataFrame()
        Y = pd.DataFrame(columns=[rising_column_name])
        tmp_row = {}
        
        df_subset = entire_dataframe
        row = self.get_technical_indicators(df_subset, ta_subset=indicators)
        X = pd.DataFrame(row)
        X.index = entire_dataframe.index
            
        return pd.concat([X, entire_dataframe], axis=1).dropna()
    
    def create_dataset_without_tai(self, entire_dataframe, past_timeframe=20, stock_price="Close", preprocess=True):
        start_index = 0
        end_index = past_timeframe
        entire_dataframe_rows_number = len(entire_dataframe)
        rising_column_name = "Rising"
        X = []
        Y = pd.DataFrame(columns=[rising_column_name])
        
        while end_index < entire_dataframe_rows_number:
            df_subset = entire_dataframe.iloc[start_index:end_index]
            if preprocess:
                X.append(preprocessing.scale(df_subset.values))
            else:
                X.append(df_subset.values)
            is_price_rising =  1 if (entire_dataframe[stock_price].iloc[end_index] > entire_dataframe[stock_price].iloc[end_index - 1]) else 0
            Y = Y.append([{rising_column_name: is_price_rising}])
            start_index+= 1
            end_index+= 1
            
        return np.array(X),Y
    
    def create_train_test_datasets(self, X,Y, train_percentage=0.8, with_tai=True):
        
        indices = list(range(0, len(X)))
#        np.random.shuffle(indices)
        last_train_index = round(len(indices)*train_percentage)
        
        if with_tai:
            X_train = X.iloc[indices[0:last_train_index]]
            Y_train = Y.iloc[indices[0:last_train_index]]
            
            X_test = X.iloc[indices[last_train_index:]]
            Y_test = Y.iloc[indices[last_train_index:]]
            
        else:
            X_train = X[indices[0:last_train_index]]
            Y_train = Y.iloc[indices[0:last_train_index]]
            
            X_test = X[indices[last_train_index:]]
            Y_test = Y.iloc[indices[last_train_index:]]
                
        
        return X_train, Y_train, X_test, Y_test
    
    def datasets_as_matrices(self, X_train, Y_train, X_test, Y_test, datasets_form="Normal", preprocess = True):
        
        X_train = X_train.values
        Y_train = Y_train.values
        X_test = X_test.values
        Y_test = Y_test.values
        
        if preprocess:
            self.scaler = preprocessing.StandardScaler().fit(X_train)
            X_train= self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        if datasets_form == "RNN":
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
#            Y_train = np.reshape(Y_train, (Y_train.shape[0], 1, Y_train.shape[1]))
#            Y_test = np.reshape(Y_test, (Y_test.shape[0], 1, Y_test.shape[1]))
            
        return X_train, Y_train, X_test, Y_test
        
