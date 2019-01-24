#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:47:41 2017

@author: squall
"""

import pandas_datareader.data as web
import pandas as pd
import numpy as np
import urllib.request as Urllib
import datetime as dt
import matplotlib.pyplot as plt

import datetime

class DataAccess:
    
    def __init__(self):
        pass
    
    def get_stocks(self, symbols, start, end, source = "google"):
        return web.DataReader(symbols, source, start, end)
    
    def get_intraday_data(self, symbols, period, days):
        url = "https://www.google.com/finance/getprices?i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}"
        url = url.format(period=period, days=days, ticker=symbols)
        response = Urllib.urlopen(url).read()
        print(url)
        
        return response
    
    def get_stock_from_csv(self, filename, start="", end="", drop_date_company = True):
        data = pd.read_csv(filename)
        data = data[(data != 0).all(1)]
        data = data.drop_duplicates()
        data = data.set_index(pd.Index.unique(pd.DatetimeIndex(data["Date"])))
        data.sort_index(inplace=True)
        if drop_date_company:
            data.drop(["Date", "Company"], axis=1, inplace=True)
        
        return data
        
    def get_multiple_stocks(self, symbols, start, end, source = "google"):
        stocks = {}
        for symbol in symbols:
            stocks[symbol] = pdr.DataReader(symbol, source, start, end)
            
        return stocks