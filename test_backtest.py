# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:48:06 2017

@author: User
"""

from datetime import datetime
import backtrader as bt
from backtrader import CommInfoBase
from my_strategy import MyStrategy
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds
from DataAccess import DataAccess


class SmaCross(bt.SignalStrategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
#        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
#        crossover = bt.ind.CrossOver(sma1, sma2)
#        self.signal_add(bt.SIGNAL_LONG, crossover)
        
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

cerebro = bt.Cerebro()

DA = DataAccess()

pd_data = DA.get_stock_from_csv("egx30 index 10y good.txt", start="", end="")
pd_data.dropna(inplace=True)
pd_data = pd_data.sort_index()
pd_data = pd_data.ix["2015-01-04":"2016-01-05"]
data2 = bt.feeds.PandasData(
    dataname=pd_data,
    datetime=None,
    high=1,
    low=2,
    open=0,
    close=3,
    volume=4,
    openinterest=None
)

data1 = btfeeds.GenericCSVData(
    dataname='tun index 10y good.txt',

#    todate=datetime(2016, 8, 10),
#    fromdate=datetime(2009, 10, 5),

    nullvalue=0.0,

    dtformat=('%Y-%m-%d'),
    tmformat=('%H.%M.%S'),

    datetime=1,
    high=3,
    low=4,
    open=2,
    close=5,
    volume=6,
    openinterest=-1,
)

data0 = bt.feeds.YahooFinanceData(dataname='googl', fromdate=datetime(2016, 1, 1),
                                  todate=datetime(2017, 2, 1))
cerebro.adddata(data2) 
#cerebro.addstrategy(SmaCross)
cerebro.addstrategy(MyStrategy)

#my_sharpe_ratio = bt.analyzers.SharpeRatio(riskfreerate=0.12)

#cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')
cerebro.addanalyzer(btanalyzers.SharpeRatio_A, _name='mysharpe_a')
cerebro.addanalyzer(btanalyzers.DrawDown, _name='mydrawdown')


cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years,
                    data=data2, _name='datareturns')

cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years, _name='timereturns')



cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.01, commtype=CommInfoBase.COMM_FIXED)
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

results = cerebro.run()


strat = results[0]

print('Sharpe Ratio:', strat.analyzers.mysharpe.get_analysis())
print('Sharpe Ratio A:', strat.analyzers.mysharpe_a.get_analysis())
print('Drawdown:', strat.analyzers.mydrawdown.get_analysis())

tret_analyzer = strat.analyzers.getbyname('timereturns')
print("my return:", tret_analyzer.get_analysis())
tdata_analyzer = strat.analyzers.getbyname('datareturns')
print("buy and hold", tdata_analyzer.get_analysis())

#pyfoliozer = strat.analyzers.getbyname('pyfolio')
#returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
#print('-- RETURNS')
#print(returns)
#print('-- POSITIONS')
#print(positions)
#print('-- TRANSACTIONS')
#print(transactions)
#print('-- GROSS LEVERAGE')
#print(gross_lev) 
#
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
#
#import pyfolio as pf
#pf.create_full_tear_sheet(
#    returns,
#    positions=positions,
#    transactions=transactions,
#    gross_lev=gross_lev,
#    round_trips=True)

cerebro.plot()

