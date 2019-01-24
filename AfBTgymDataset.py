# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:39:40 2017

@author: User
"""

import os
from btgym import BTgymDataset
from DataAccess import DataAccess

class AfBTgymDataset(BTgymDataset):
    
    def read_csv(self, filename=None):
        """
        Populates instance by loading data: CSV file --> pandas dataframe
        """
#        print("{}".format(filename))
        if filename:
#            print("Yes, filename")
            self.filename = filename  # override data source if one is given
        try:
            assert self.filename and os.path.isfile(self.filename)
            DA = DataAccess()
            self.data = DA.get_stock_from_csv(self.filename, start="", end="")
            self.data.dropna(inplace=True)
            self.log.info('Loaded {} records from <{}>.'.format(self.data.shape[0], self.filename))

        except:
            try:
                assert 'episode_dataset' in self.filename
                self.log.warning('Attempt to load data into episode dataset: ignored.')
                return None

            except:
                msg = 'Data file <{}> not specified / not found.'.format(str(self.filename))
                self.log.error(msg)