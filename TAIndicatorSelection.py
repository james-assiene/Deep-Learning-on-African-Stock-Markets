# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:31:59 2017

@author: assie
"""

from DataAccess import DataAccess
from DataPreprocessing import DataPreprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from sklearn import preprocessing


import threading
import talib as ta

class TAIndicatorSelection:
    
    def __init__(self):
        self.DA = DataAccess()
        self.DP = DataPreprocessing()

        self.value_of_interest = "Close"
        
        print("all tai")
        
        self.create_all_ta_indicators()
        
        print("done")
        
    def create_all_ta_indicators(self):
        self.all_ta_indicators = []
        
        self.time_periods_range = range(1, 101)
        self.slow_periods_range = range(1, 101)
        self.fast_periods_range = range(1,31)
        self.signal_periods_range = range(1, 101)
        
        import numpy as np
        
        self.fast_limit_range = np.linspace(0, 1, 11)
        self.slow_limit_range = np.linspace(0, 1, 11)
        
        self.acceleration_range = np.linspace(0, 0.1, 11)
        self.maximum_range = np.linspace(0, 1, 11)
        
        self.v_factor_range = np.linspace(0, 1, 11)
        self.nb_dev_up_down = range(1, 11)
        
        self.ma_type_range = [ta.MA_Type.EMA, ta.MA_Type.WMA, ta.MA_Type.DEMA, ta.MA_Type.KAMA, ta.MA_Type.MAMA, ta.MA_Type.SMA, ta.MA_Type.T3, ta.MA_Type.TEMA]
        

        #VOLUMNE_INDICATORS
        self.all_ta_indicators+= ([{"name": "AD", "voi": "Close", "params": {}}])
        self.all_ta_indicators+= ([{"name": "ADOSC", "voi": "Close", "params": {"fastperiod": f_val, "slowperiod": s_val}} for f_val in self.fast_periods_range for s_val in self.slow_periods_range])
        self.all_ta_indicators+= ([{"name": "OBV", "voi": "Close", "params": {}}])
        
        #VOLATILITY_INDICATORS
        self.all_ta_indicators+= ([{"name": "ATR", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "TRANGE", "voi": "Close", "params": {}}])
        
        #PATTERN_RECOGNITION
        self.all_ta_indicators+= ([{"name": "CDLDRAGONFLYDOJI", "voi": "Close", "params": {}}])
        
        #OVERLAP_STUDIES
        self.all_ta_indicators+= ([{"name": "SMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "EMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "DEMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "KAMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "TEMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "TRIMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "WMA", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "MIDPOINT", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "MA", "voi": "Close", "params": {"timeperiod": 3, "matype": ta.MA_Type.KAMA}}])
        self.all_ta_indicators+= ([{"name": "MAMA", "voi": "Close", "params": {"fastlimit": fl_val, "slowlimit": sl_val}} for fl_val in self.fast_limit_range for sl_val in self.slow_limit_range]) #bug
        self.all_ta_indicators+= ([{"name": "SAR", "voi": "Close", "params": {"acceleration": a_val, "maximum": m_val}} for a_val in self.acceleration_range for m_val in self.maximum_range]) 
        self.all_ta_indicators+= ([{"name": "SAREXT", "voi": "Close", "params": {}}]) #there's a lot of optionnal parameters
        self.all_ta_indicators+= ([{"name": "T3", "voi": "Close", "params": {"timeperiod": n_val, "vfactor": v_val}} for n_val in self.time_periods_range for v_val in self.v_factor_range]) #bug
        #self.all_ta_indicators+= ([{"name": "MIDPRICE", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range]) #bug
        #self.all_ta_indicators+= ([{"name": "MAVP", "voi": "Close", "params": {"minperiod": 2, "maxperiod": 30, "periods": np.array([float(x) for x in [1,2,3]])}}]) #input lengths are different
        #self.all_ta_indicators+= ([{"name": "HT_TRENDLINE", "voi": "Close", "params": {}}])
        self.all_ta_indicators+= ([{"name": "BBANDS", "voi": "Close", "params": {"timeperiod": n_val, "nbdevup": nb_up, "nbdevdn": nb_down, "matype": ma_type}} for n_val in self.time_periods_range for nb_up in self.nb_dev_up_down for nb_down in self.nb_dev_up_down for ma_type in self.ma_type_range])
        #MOMENTUM
        self.all_ta_indicators+= ([{"name": "MOM", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "ROC", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "RSI", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "TRIX", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "WILLR", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "ULTOSC", "voi": "Close", "params": {"timeperiod1": n1_val, "timeperiod2": n2_val, "timeperiod3": n3_val}} for n1_val in self.time_periods_range for n2_val in self.time_periods_range for n3_val in self.time_periods_range])
        #self.all_ta_indicators+= ([{"name": "CMO", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range]) #bug
        self.all_ta_indicators+= ([{"name": "ADX", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "CCI", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "MINUS_DI", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "PLUS_DI", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "DX", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "AROON", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "AROONOSC", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "MINUS_DM", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "PLUS_DM", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "BOP", "voi": "Close", "params": {}}])
        self.all_ta_indicators+= ([{"name": "MFI", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "ADXR", "voi": "Close", "params": {"timeperiod": n_val}} for n_val in self.time_periods_range])
        self.all_ta_indicators+= ([{"name": "APO", "voi": "Close", "params": {"slowperiod": s_val, "fastperiod": f_val, "matype": ma_type}} for s_val in self.slow_periods_range for f_val in self.fast_periods_range for ma_type in self.ma_type_range]) #bug
        self.all_ta_indicators+= ([{"name": "PPO", "voi": "Close", "params": {"slowperiod": s_val, "fastperiod": f_val, "matype": ma_type}} for s_val in self.slow_periods_range for f_val in self.fast_periods_range for ma_type in self.ma_type_range]) #bug
        self.all_ta_indicators+= ([{"name": "MACD", "voi": "Close", "params": {"slowperiod": s_val, "fastperiod": f_val, "signalperiod": sig_val}} for s_val in self.slow_periods_range for f_val in self.fast_periods_range for sig_val in self.signal_periods_range]) #bug
#        self.all_ta_indicators+= ([{"name": "MACDEXT", "voi": "Close", "params": {"slowperiod": s_val,"fastperiod": f_val, "signalperiod": sig_val, "slowmatype": s_ma_type, "fastmatype": f_ma_type, "signalmatype": sig_ma_type}} for s_val in self.slow_periods_range for f_val in self.fast_periods_range for sig_val in self.signal_periods_range for s_ma_type in self.ma_type_range for f_ma_type in self.ma_type_range for sig_ma_type in self.ma_type_range]) #bug
#        self.all_ta_indicators+= ([{"name": "STOCH", "voi": "Close", "params": {"slowk_period": sk_val, "fastk_period": fk_val, "slowd_period": sd_val, "slowd_matype": sd_ma_type, "slowk_matype": sk_ma_type}} for sk_val in self.slow_periods_range for sd_val in self.slow_periods_range for fk_val in self.fast_periods_range for sd_ma_type in self.ma_type_range for sk_ma_type in self.ma_type_range]) #bug
#        self.all_ta_indicators+= ([{"name": "STOCHF", "voi": "Close", "params": {"fastd_period": fd_val, "fastk_period": fk_val, "fastd_matype": fd_ma_type}} for fk_val in self.fast_periods_range for fd_val in self.fast_periods_range for fd_ma_type in self.ma_type_range]) #bug
#        #self.all_ta_indicators+= ([{"name": "STOCHRSI", "voi": "Close", "params": {"fastd_period": 12, "fastk_period": 12, "fastd_matype": ta.MA_Type.EMA, "timeperiod": 3}}]) #bug
        print("i'm done")
        return self.all_ta_indicators
    
    
    def greedy_forward(self, data, model="LR", past_timeframe=100):
        return self.greedy_forward_mp(data, model, past_timeframe)
    
    
    def greedy_forward_sp(self, data, model="LR", past_timeframe=100):
        
        best_ta_indicators = []
        final_ta_indicators = []
        greedy_features = []
        best_mean = 0
        best_means = []
        best_std = 0
        results = []
        current_features_number = 0
        add_feature_to_final_ta = False
        
        possible_features = self.all_ta_indicators[0:5]
        feature_to_remove_index = 0
        
        
        
        while len(possible_features) > 0:
            best_ta_indicators.append([{}])
            results.append([])
            best_means.append(0)
            greedy_features.append({})
            
            for feature_index, feature in enumerate(possible_features):
                
                scores_mean, scores_std, indicators_names, feature_index = self.evaluate_one_features_set(data, past_timeframe, greedy_features, model, feature, feature_index, current_features_number)
                
                results[current_features_number].append({indicators_names: [scores_mean, scores_std]})
                        
                if scores_mean > best_means[current_features_number]:
                    best_means[current_features_number] = scores_mean                    
                    feature_to_remove_index = feature_index
                    
                    if scores_mean > best_mean:
                        best_mean = scores_mean
                        add_feature_to_final_ta = True
                    
                        
            if add_feature_to_final_ta:
                final_ta_indicators.append(possible_features[feature_to_remove_index])
                add_feature_to_final_ta = False
            greedy_features[current_features_number] = possible_features[feature_to_remove_index]
            best_ta_indicators[current_features_number] = deepcopy(greedy_features)
            del(possible_features[feature_to_remove_index])
            current_features_number+= 1
        
        return final_ta_indicators, best_ta_indicators, results
    
    
    def greedy_forward_mt(self, data, model="LR", past_timeframe=100):
        
        best_ta_indicators = []
        final_ta_indicators = []
        greedy_features = []
        best_mean = 0
        best_means = []
        best_std = 0
        results = []
        current_features_number = 0
        add_feature_to_final_ta = False
        
        possible_features = self.all_ta_indicators[0:5]
        feature_to_remove_index = 0
        
        
        
        while len(possible_features) > 0:
            best_ta_indicators.append([{}])
            results.append([])
            best_means.append(0)
            greedy_features.append({})
            result = []
            
            feature_start_index = 0
            max_number_of_processes = 100
            number_of_features = len(possible_features)
            
            print("here I am")
            
            while feature_start_index < number_of_features:
                
                print("Trouble... {0}".format(feature_start_index))
                
                result = []
                    
                number_of_processes = max_number_of_processes if feature_start_index + max_number_of_processes <= number_of_features else number_of_features - feature_start_index
                
                pool = ThreadPool(processes=1)
                
                params = [(data, past_timeframe, greedy_features, model, possible_features[feature_start_index + i], feature_start_index + i, current_features_number) for i in range(number_of_processes)]
                
                async_results = [pool.apply_async(self.evaluate_one_features_set, params[i]) for i in range(number_of_processes)]
                
                for i in range(number_of_processes):
                    result.append(async_results[i].get())
                
                feature_start_index+= max_number_of_processes
                
                for (scores_mean, scores_std, indicators_names, feature_index) in result:
                    
                    results[current_features_number].append({indicators_names: [scores_mean, scores_std]})
                        
                    if scores_mean > best_means[current_features_number]:
                        best_means[current_features_number] = scores_mean                    
                        feature_to_remove_index = feature_index
                        
                        if scores_mean > best_mean:
                            best_mean = scores_mean
                            add_feature_to_final_ta = True
                        
            if add_feature_to_final_ta:
                final_ta_indicators.append(possible_features[feature_to_remove_index])
                add_feature_to_final_ta = False
            greedy_features[current_features_number] = possible_features[feature_to_remove_index]
            best_ta_indicators[current_features_number] = deepcopy(greedy_features)
            del(possible_features[feature_to_remove_index])
            current_features_number+= 1
        
        return final_ta_indicators, best_ta_indicators, results
    
        
    def greedy_forward_mp(self, data, model="LR", past_timeframe=100):
        
        best_ta_indicators = []
        final_ta_indicators = []
        greedy_features = []
        best_mean = 0
        best_means = []
        best_std = 0
        results = []
        current_features_number = 0
        add_feature_to_final_ta = False
        
        possible_features = self.all_ta_indicators
        feature_to_remove_index = 0
        
        print("step 1")
        
        while len(possible_features) > 0:
            print("step 2")
            best_ta_indicators.append([{}])
            results.append([])
            best_means.append(0)
            greedy_features.append({})
            result = []
            
            feature_start_index = 0 
            max_number_of_processes = 4
            number_of_features = len(possible_features)
            
            print("here I am")
            
            while feature_start_index < number_of_features:
                
                print("Trouble... {0}".format(feature_start_index))
                
                result = []
                    
                number_of_processes = max_number_of_processes if feature_start_index + max_number_of_processes <= number_of_features else number_of_features - feature_start_index
                
                pool = Pool(processes=number_of_processes)
                
                params = [(data, past_timeframe, greedy_features, model, possible_features[feature_start_index + i], feature_start_index + i, current_features_number) for i in range(number_of_processes)]
                
                result = pool.map(self.multi_run_wrapper_evaluate_one_features_set, params)
                
                pool.close()
                
                feature_start_index+= max_number_of_processes
                
                for (scores_mean, scores_std, indicators_names, feature_index) in result:
                    
                    results[current_features_number].append({indicators_names: [scores_mean, scores_std]})
                        
                    if scores_mean > best_means[current_features_number]:
                        best_means[current_features_number] = scores_mean                    
                        feature_to_remove_index = feature_index
                        
                        if scores_mean > best_mean:
                            best_mean = scores_mean
                            add_feature_to_final_ta = True
                        
            if add_feature_to_final_ta:
                final_ta_indicators.append(possible_features[feature_to_remove_index])
                add_feature_to_final_ta = False
            greedy_features[current_features_number] = possible_features[feature_to_remove_index]
            best_ta_indicators[current_features_number] = deepcopy(greedy_features)
            del(possible_features[feature_to_remove_index])
            current_features_number+= 1
        
        return final_ta_indicators, best_ta_indicators, results
    
    def foo(self, x):
        print("hello  {0}".format(x))
        
        return (x ** 2, x ** 2, x ** 2, x ** 2)
    
    
        
    def fake_greedy_forward_mp(self, data, model="LR", past_timeframe=100):
        
        best_ta_indicators = []
        final_ta_indicators = []
        greedy_features = []
        best_mean = 0
        best_means = []
        best_std = 0
        results = []
        current_features_number = 0
        add_feature_to_final_ta = False
        
        possible_features = self.all_ta_indicators
        feature_to_remove_index = 0
        
        
        
        while len(possible_features) > 0:
            best_ta_indicators.append([{}])
            results.append([])
            best_means.append(0)
            greedy_features.append({})
            result = []
            
            feature_start_index = 0 
            max_number_of_processes = 8
            number_of_features = len(possible_features)
            
            print("here I am")
            
            while feature_start_index < number_of_features:
                
                print("Trouble... {0}".format(feature_start_index))
                
                result = []
                    
                number_of_processes = max_number_of_processes if feature_start_index + max_number_of_processes <= number_of_features else number_of_features - feature_start_index
                
                pool = Pool(processes=number_of_processes)
                
                params = [i for i in range(number_of_processes)]
                
                result = pool.map(self.foo, params)
                
                feature_start_index+= max_number_of_processes
                
                for (scores_mean, scores_std, indicators_names, feature_index) in result:
                    
                    results[current_features_number].append({indicators_names: [scores_mean, scores_std]})
                        
                    if scores_mean > best_means[current_features_number]:
                        best_means[current_features_number] = scores_mean                    
                        feature_to_remove_index = feature_index
                        
                        if scores_mean > best_mean:
                            best_mean = scores_mean
                            add_feature_to_final_ta = True
                        
            if add_feature_to_final_ta:
                final_ta_indicators.append(possible_features[feature_to_remove_index])
                add_feature_to_final_ta = False
            greedy_features[current_features_number] = possible_features[feature_to_remove_index]
            best_ta_indicators[current_features_number] = deepcopy(greedy_features)
            del(possible_features[feature_to_remove_index])
            current_features_number+= 1
        
        return final_ta_indicators, best_ta_indicators, results
    
    def multi_run_wrapper_evaluate_one_features_set(self, args):
        return self.evaluate_one_features_set(*args)
    
    
    def evaluate_one_features_set(self, data, past_timeframe, greedy_features, model, feature, feature_index, current_features_number):
        
        print("look at me now!")
        
        ta_indicators = deepcopy(greedy_features)
        ta_indicators[current_features_number] = feature
        X,Y = self.DP.create_dataset(data, past_timeframe=100, indicators=ta_indicators)
        self.scaler = preprocessing.StandardScaler().fit(X.values)
        X = self.scaler.transform(X.values)
                
        clf = ""
                
        if model == "SVM":
            clf = svm.SVC()
                    
        elif model == "RF":
            clf = RandomForestClassifier()
                    
        elif model == "LR":
            clf = LogisticRegression(n_jobs=-1)
                    
        print("before cv")
        from sklearn.externals.joblib import Parallel, parallel_backend
        with parallel_backend("threading"):
            scores = cross_val_score(clf, X, Y.values.reshape(len(Y),), cv=5, n_jobs=-1)
        print("after cv")
        scores_mean = scores.mean()
        scores_std = scores.std()
        
        indicators_names = "".join([indicator["name"] + "," for indicator in ta_indicators])
        print("indicators : {0}, mean : {1}, std : {2}".format(indicators_names, scores_mean, scores_std))
        
        return scores_mean, scores_std, indicators_names, feature_index
        
    
        
    