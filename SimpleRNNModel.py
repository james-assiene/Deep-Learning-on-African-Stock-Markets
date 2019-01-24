#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:52:01 2017

@author: squall
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.np_utils import to_categorical

class SimpleRNNModel:
    
    def __init__(self, num_samples, num_features):
        self.num_features = num_features
        self.num_samples = num_samples
        print("")
        
    def __call__(self, optimizer='rmsprop', init='glorot_uniform', neurons=128):
        
        simple_rnn_model = Sequential()
        simple_rnn_model.add(SimpleRNN(units=neurons,
                       input_shape=(1, self.num_features), kernel_initializer=init))  # returns a sequence of vectors of dimension 32
        #simple_rnn_model.add(simple_rnn(32))  # returns a sequence of vectors of dimension 32
        #simple_rnn_model.add(simple_rnn(32))  # return a single vector of dimension 32
        simple_rnn_model.add(Dense(2, activation='softmax'))
        
        simple_rnn_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        return simple_rnn_model