# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:32:44 2017

@author: User
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras import optimizers, initializers
from keras.layers.normalization import BatchNormalization

class LSTMNoTaiModel:
    
    def __init__(self, num_samples, num_features, sequence_length):
        self.num_features = num_features
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        print("")
        
    def __call__(self, optimizer='adam', init='glorot_normal'):
        
        kernel_size = 5
        filters = 75
        adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = optimizers.SGD(lr=10000, momentum=0.0, decay=0.0, nesterov=False)
        random_normal = initializers.RandomNormal(mean=0.0, stddev=0.0005, seed=None)
        
        lstm_model = Sequential()
        lstm_model.add(LSTM(filters, input_shape=(self.sequence_length, self.num_features), kernel_initializer="glorot_normal"))
        lstm_model.add(Activation('relu'))
        lstm_model.add(Dense(units=2))
        lstm_model.add(Activation('softmax'))
        
        lstm_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
        
        return lstm_model