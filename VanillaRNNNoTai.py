# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:07:21 2017

@author: User
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.recurrent import SimpleRNN
from keras import optimizers, initializers
from keras.layers.normalization import BatchNormalization

class VanillaRNNNoTaiModel:
    
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
        
        vanilla_rnn_model = Sequential()
        vanilla_rnn_model.add(SimpleRNN(filters, input_shape=(self.sequence_length, self.num_features), kernel_initializer="glorot_normal"))
#        vanilla_rnn_model.add(Dropout(0.5))
        vanilla_rnn_model.add(Activation('relu'))
        vanilla_rnn_model.add(Dense(units=2))
        vanilla_rnn_model.add(Activation('softmax'))
        
        vanilla_rnn_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
        
        return vanilla_rnn_model