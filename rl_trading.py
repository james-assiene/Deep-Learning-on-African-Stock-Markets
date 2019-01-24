# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:16:45 2017

@author: User
"""

import backtrader as bt
from btgym import BTgymDataset, BTgymStrategy, BTgymEnv
from DataAccess import DataAccess
from AfBTgymDataset import AfBTgymDataset
import IPython.display as Display
import PIL.Image as Image
import numpy as np
import random
 
def show_rendered_image(rgb_array):
    """
    Convert numpy array to RGB image using PILLOW and
    show it inline using IPykernel.
    """
    Display.display(Image.fromarray(rgb_array))

def render_all_modes(env):
    """
    Retrieve and show environment renderings
    for all supported modes.
    """
    for mode in env.metadata['render.modes']:
        print('[{}] mode:'.format(mode))
        show_rendered_image(env.render(mode))

def take_some_steps(env, some_steps):
    """Just does it. Acting randomly."""
    print("here i am")
    for step in range(some_steps):
        rnd_action = int((env.action_space.n)*random.random())
        o, r, d, i = env.step(rnd_action)
        if d:
            print('Episode finished,')
            break
    print(step+1, 'actions made.\n')
    
def under_the_hood(env):
    """Shows environment internals."""
    for attr in ['dataset','strategy','engine','renderer','network_address']:
        print ('\nEnv.{}: {}'.format(attr, getattr(env, attr)))

    for params_name, params_dict in env.params.items():
        print('\nParameters [{}]: '.format(params_name))
        for key, value in params_dict.items():
            print('{} : {}'.format(key,value))

MyCerebro = bt.Cerebro()
MyCerebro.addstrategy(BTgymStrategy,
                      state_shape=(20,4),
                      skip_frame=5,
                      state_low=None,
                      state_high=None,
                      drawdown_call=50,
                      )
 
MyCerebro.broker.setcash(100.0)
MyCerebro.broker.setcommission(commission=0.001)
MyCerebro.addsizer(bt.sizers.SizerFix, stake=10)
MyCerebro.addanalyzer(bt.analyzers.DrawDown)

DA = DataAccess()

pd_data = DA.get_stock_from_csv("egx30 index 10y good.txt", start="", end="")
pd_data.dropna(inplace=True)
#pd_data = pd_data.ix["2015-01-01":]
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
 
MyDataset = AfBTgymDataset(filename="egx30 index 10y good.csv",
     start_weekdays=[0, 1, 2, 3, 4], 
     episode_len_days=252, 
     timeframe=60*24,
     time_gap_days=115,
     datetime=None,
    high=1,
    low=2,
    open=0,
    close=3,
    volume=4,
    openinterest=None
                         )
 
env = BTgymEnv(dataset=MyDataset,
                         engine=MyCerebro,
                         port=5555,
                         verbose=1,
                         )

print("let's go")
#env.reset()
take_some_steps(env, 100)
render_all_modes(env)
