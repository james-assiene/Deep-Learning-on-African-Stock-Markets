#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:41:47 2017

@author: squall
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.np_utils import to_categorical

class MLPModel:
    
    def __init__(self, num_samples, num_features):
        self.num_features = num_features
        self.num_samples = num_samples
        print("")
        
    def __call__(self, optimizer='rmsprop', init='glorot_uniform', neurons=220):
        
        mlp_model = Sequential()
        mlp_model.add(Dense(units=neurons, input_dim=self.num_features, kernel_initializer=init))
        mlp_model.add(Activation('relu'))
        mlp_model.add(Dense(units=2))
        mlp_model.add(Activation('softmax'))
        
        mlp_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        return mlp_model