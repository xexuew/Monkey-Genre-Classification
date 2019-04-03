#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 22:31:22 2018

@author: josetorronteras
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras import regularizers
from keras import backend as K

class CNNModel(object):

    def __init__(self, config, X):
        self.NB_FILTERS = int(config['CNN_CONFIGURATION']['NB_FILTERS']) 
        self.NB2_FILTERS = int(config['CNN_CONFIGURATION']['NB2_FILTERS'])
        self.CONV1_SIZE = int(config['CNN_CONFIGURATION']['CONV1_SIZE'])
        self.CONV2_SIZE = int(config['CNN_CONFIGURATION']['CONV2_SIZE'])
        self.NUM_CLASSES = int(config['CNN_CONFIGURATION']['NUM_CLASSES'])
        self.INPUT_SHAPE = X[1].shape
        
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(self.NB_FILTERS, (self.CONV1_SIZE , self.CONV1_SIZE ), activation='relu', input_shape=self.INPUT_SHAPE))
        model.add(Conv2D(self.NB_FILTERS, (self.CONV1_SIZE, self.CONV1_SIZE), activation='relu'))
        model.add(MaxPooling2D(pool_size=(self.CONV2_SIZE , self.CONV2_SIZE )))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(self.NB2_FILTERS, (self.CONV1_SIZE, self.CONV1_SIZE), activation='relu'))
        model.add(Conv2D(self.NB2_FILTERS, (self.CONV1_SIZE, self.CONV1_SIZE), activation='relu'))
        model.add(MaxPooling2D(pool_size=(self.CONV2_SIZE , self.CONV2_SIZE )))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        
        return model
        
        