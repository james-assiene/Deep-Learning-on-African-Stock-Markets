#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 22:47:43 2017

@author: squall
"""

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
plt.savefig('vrx.png')

plt.show()