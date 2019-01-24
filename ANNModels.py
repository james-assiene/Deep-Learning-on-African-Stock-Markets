#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:00:32 2017

@author: squall
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.np_utils import to_categorical

class ANNModels:
    
    def __init__(self, num_samples, num_features):
        self.num_features = num_features
        self.num_samples = num_samples
        print("")
        
    def create_mlp_model(self):
        
        mlp_model = Sequential()
        mlp_model.add(Dense(units=220, input_dim=self.num_features))
        mlp_model.add(Activation('relu'))
        mlp_model.add(Dense(units=2))
        mlp_model.add(Activation('softmax'))
        
        mlp_model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        return mlp_model
    
    def create_simple_rnn_model(self):
        
        simple_rnn_model = Sequential()
        simple_rnn_model.add(LSTM(128,
                       input_shape=(1, self.num_features)))  # returns a sequence of vectors of dimension 32
       
        simple_rnn_model.add(Dense(2, activation='softmax'))
        
        simple_rnn_model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        return simple_rnn_model
    
    def create_lstm_model(self):
        
        lstm_model = Sequential()
        lstm_model.add(SimpleRNN(128,
                       input_shape=(1, self.num_features)))  # returns a sequence of vectors of dimension 32
        #lstm_model.add(LSTM(32))  # returns a sequence of vectors of dimension 32
        #lstm_model.add(LSTM(32))  # return a single vector of dimension 32
        lstm_model.add(Dense(2, activation='softmax'))
        
        lstm_model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        return lstm_model
    
    def create_conv1d_model(self):
        
        conv_model = Sequential()
        conv_model.add(Conv1D(64, 3, activation='relu', input_shape=(self.num_samples, self.num_features)))
        conv_model.add(Conv1D(64, 3, activation='relu'))
        conv_model.add(MaxPooling1D(3))
        conv_model.add(Conv1D(128, 3, activation='relu'))
        conv_model.add(Conv1D(128, 3, activation='relu'))
        conv_model.add(GlobalAveragePooling1D())
        conv_model.add(Dropout(0.5))
        conv_model.add(Dense(2, activation='sigmoid'))
        
        conv_model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        return conv_model