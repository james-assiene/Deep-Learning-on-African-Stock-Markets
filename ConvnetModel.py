#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:13:18 2017

@author: squall
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras import optimizers, initializers, regularizers

class ConvNetModel:
    
    def __init__(self, num_samples, num_features, sequence_length):
        self.num_features = num_features
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        print("")
        
    def __call__(self, optimizer='adam', init='glorot_normal', filters=64, kernel_size=5):
        
        adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = optimizers.SGD(lr=10000, momentum=0.0, decay=0.0, nesterov=False)
        random_normal = initializers.RandomNormal(mean=0.0, stddev=0.0005, seed=None)
        
        convnet_model = Sequential()
        convnet_model.add(Conv1D(filters, kernel_size, input_shape=(self.sequence_length, self.num_features), kernel_initializer="glorot_normal", activity_regularizer=regularizers.l1(0.01)))
        convnet_model.add(Activation('relu'))
#        convnet_model.add(Conv1D(filters, kernel_size, kernel_initializer="glorot_normal"))
#        convnet_model.add(Activation('relu'))
#        convnet_model.add(MaxPooling1D(2))
#        convnet_model.add(Dropout(0.3))
        convnet_model.add(Flatten())
        convnet_model.add(Dense(units=2))
        convnet_model.add(Activation('softmax'))
        
        convnet_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
        
        return convnet_model
    

class ConvNetModel2L:
    
    def __init__(self, num_samples, num_features, sequence_length):
        self.num_features = num_features
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        print("")
        
    def __call__(self, optimizer='adam', init='glorot_normal', filters=[64, 64], kernel_size = [5,3]):
        
        adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = optimizers.SGD(lr=10000, momentum=0.0, decay=0.0, nesterov=False)
        random_normal = initializers.RandomNormal(mean=0.0, stddev=0.0005, seed=None)
        
        convnet_model = Sequential()
        convnet_model.add(Conv1D(filters[0], kernel_size[0], input_shape=(self.sequence_length, self.num_features), kernel_initializer="glorot_normal"))
        convnet_model.add(Activation('relu'))
        convnet_model.add(Conv1D(filters[1], kernel_size[1], kernel_initializer="glorot_normal"))
        convnet_model.add(Activation('relu'))
        convnet_model.add(MaxPooling1D(2))
#        convnet_model.add(Dropout(0.3))
        convnet_model.add(Flatten())
        convnet_model.add(Dense(units=2))
        convnet_model.add(Activation('softmax'))
        
        convnet_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
        
        return convnet_model
    
class ConvNetModelNL:
    
    def __init__(self, num_samples, num_features, sequence_length):
        self.num_features = num_features
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        print("")
        
    def __call__(self, conv_num, conv_pool_num, fc_num, filters, kernel_size, fc_size, pool_size, optimizer='adam', init='glorot_normal'):
        
        adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = optimizers.SGD(lr=10000, momentum=0.0, decay=0.0, nesterov=False)
        random_normal = initializers.RandomNormal(mean=0.0, stddev=0.0005, seed=None)
        
        convnet_model = Sequential()
        convnet_model.add(Conv1D(filters[0], kernel_size[0], padding="same", kernel_initializer="glorot_normal", input_shape=(self.sequence_length, self.num_features)))
        convnet_model.add(Activation('relu'))
        for m in range(conv_pool_num):
            for n in range(1, conv_num):
                convnet_model.add(Conv1D(filters[n], kernel_size[n], kernel_initializer="glorot_normal", padding="same"))
                convnet_model.add(Activation('relu'))
                
                
            convnet_model.add(MaxPooling1D(pool_size[m]))
            
        convnet_model.add(Flatten())
        
        for k in range(fc_num):
#            , kernel_regularizer=regularizers.l1(0.01)
            convnet_model.add(Dense(units=fc_size[k]))
            convnet_model.add(Activation('relu'))
        convnet_model.add(Dropout(0.3))
        convnet_model.add(Dense(units=2))
        convnet_model.add(Activation('softmax'))
        
        convnet_model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
        
        return convnet_model