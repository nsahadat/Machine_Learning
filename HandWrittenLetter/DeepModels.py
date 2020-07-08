# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:45:27 2020

@author: mnsah
"""
#import keras
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras import models
#from keras import backend as K
#K.clear_session()

def simple1DModel(inputs, outputshape):
    
#    inputs = Input(shape=(8000,1))
    net = models.Sequential()
    #First Conv1D layer
    net.add(Conv1D(8,13, padding='valid', activation='relu', strides=1, input_shape=inputs))
    net.add(MaxPooling1D(3))
    net.add(Dropout(0.3))
    
    #Second Conv1D layer
    net.add(Conv1D(16, 11, padding='valid', activation='relu', strides=1))
    net.add(MaxPooling1D(3))
    net.add(Dropout(0.3))
    
    #Third Conv1D layer
    net.add(Conv1D(32, 9, padding='valid', activation='relu', strides=1))
    net.add(MaxPooling1D(3))
    net.add(Dropout(0.3))
    
    #Fourth Conv1D layer
    net.add(Conv1D(64, 7, padding='valid', activation='relu', strides=1))
    net.add(MaxPooling1D(3))
    net.add(Dropout(0.3))
    
    #Flatten layer
    net.add(Flatten())
    
    #Dense Layer 1
    net.add(Dense(256, activation='relu'))
    net.add(Dropout(0.3))
    
    #Dense Layer 2
    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.3))
    
    net.add(Dense(units = outputshape, activation='softmax'))
    
    net.summary()
    
    return net

def simple2DModel(inputs, outputshape):
    
#    inputs = Input(shape=(8000,1))
    net = models.Sequential()
    #First Conv1D layer
    net.add(Conv2D(32,(3,3), padding='same', activation='relu', strides=(1, 1), input_shape=inputs))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Second Conv1D layer
    net.add(Conv2D(64, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Third Conv1D layer
    net.add(Conv2D(128, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Fourth Conv1D layer
#    net.add(Conv2D(256, (3,3), padding='same', activation='relu', strides=(1, 1)))
#    net.add(MaxPooling2D(pool_size = (2,2)))
#    net.add(Dropout(0.3))
    
    #Flatten layer
    net.add(Flatten())
    
    #Dense Layer 1
    net.add(Dense(256, activation='relu'))
    net.add(Dropout(0.3))
    
    #Dense Layer 2
    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.3))
    
    net.add(Dense(units = outputshape, activation='softmax'))
    
    net.summary()
    
    return net

def simple2DModel00(inputs, outputshape):
    
#    inputs = Input(shape=(8000,1))
    net = models.Sequential()
    #First Conv1D layer
    net.add(Conv2D(64,(3,3), padding='same', activation='relu', strides=(1, 1), input_shape=inputs))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Second Conv1D layer
    net.add(Conv2D(128, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Third Conv1D layer
    net.add(Conv2D(256, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Fourth Conv1D layer
#    net.add(Conv2D(512, (3,3), padding='same', activation='relu', strides=(1, 1)))
#    net.add(MaxPooling2D(pool_size = (2,2)))
#    net.add(Dropout(0.3))
    
    #Flatten layer
    net.add(Flatten())
    
    #Dense Layer 1
    net.add(Dense(256, activation='relu'))
    net.add(Dropout(0.3))
    
    #Dense Layer 2
    net.add(Dense(128, activation='relu'))
    net.add(Dropout(0.3))
    
    net.add(Dense(units = outputshape, activation='softmax'))
    
    net.summary()
    
    return net

def simple2DModel01(inputs, outputshape):
    
#    inputs = Input(shape=(8000,1))
    net = models.Sequential()
    #First Conv1D layer
    net.add(Conv2D(32,(3,3), padding='same', activation='relu', strides=(1, 1), input_shape=inputs))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Second Conv1D layer
    net.add(Conv2D(64, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Third Conv1D layer
    net.add(Conv2D(128, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Fourth Conv1D layer
    net.add(Conv2D(256, (3,3), padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(Dropout(0.3))
    
    #Flatten layer
    net.add(Flatten())
    
    #Dense Layer 1
#    net.add(Dense(256, activation='relu'))
#    net.add(Dropout(0.3))
    
    #Dense Layer 2
#    net.add(Dense(32, activation='relu'))
#    net.add(Dropout(0.3))
    
    net.add(Dense(units = outputshape, activation='softmax'))
    
    net.summary()
    
    return net

def LeNet5(inputs, outputshape):
    net = models.Sequential()
    
    #First Conv1D layer
    # C1 Convolutional Layer
    net.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=inputs, padding='same'))
    
    # S2 Pooling Layer
    net.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    
    # C3 Convolutional Layer
    net.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    
    # S4 Pooling Layer
    net.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    
    # C5 Fully Connected Convolutional Layer
    net.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    #Flatten the CNN output so that we can connect it with fully connected layers
    net.add(Flatten())
    
    # FC6 Fully Connected Layer
    net.add(Dense(84, activation='tanh'))
    
    #Output Layer with softmax activation
    net.add(Dense(outputshape, activation='softmax'))
    
    net.summary()
    return net

def LeNetMod5(inputs, outputshape):
    dropout = 0.4
    k_size = (5,5)
    net = models.Sequential()
    
    #First Conv1D layer
    # C1 Convolutional Layer
    net.add(Conv2D(6, kernel_size=k_size, strides=(1, 1), activation='relu', input_shape=inputs, padding='same'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    # C3 Convolutional Layer
    net.add(Conv2D(16, kernel_size=k_size, strides=(1, 1), activation='relu', padding='valid'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    # C5 Fully Connected Convolutional Layer
    net.add(Conv2D(120, kernel_size=k_size, strides=(1, 1), activation='relu', padding='valid'))
    net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    #Flatten the CNN output so that we can connect it with fully connected layers
    net.add(Flatten())
    
    # FC6 Fully Connected Layer
    net.add(Dense(84, activation='relu'))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    #Output Layer with softmax activation
    net.add(Dense(outputshape, activation='softmax'))
    
    net.summary()
    return net

def VGG16(inputs, outputshape):
    net = models.Sequential()
    net.add(Conv2D(input_shape=inputs,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    net.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    net.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    net.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    net.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    net.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    net.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
    net.add(Flatten())
    net.add(Dense(units=4096,activation="relu"))
    net.add(Dense(units=4096,activation="relu"))
    net.add(Dense(units=outputshape, activation="softmax"))
    
    net.summary()
    return net

def simple2DModel02(inputs, outputshape):
    dropout = 0.5
    k_size = (5,5)
#    inputs = Input(shape=(8000,1))
    net = models.Sequential()
    #First Conv2D layer
    net.add(Conv2D(32, k_size, padding='same', activation='relu', strides=(1, 1), input_shape=inputs))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    #Second Conv2D layer
    net.add(Conv2D(64, k_size, padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    #Third Conv2D layer
    net.add(Conv2D(128, k_size, padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    #Fourth Conv2D layer
    net.add(Conv2D(256, k_size, padding='same', activation='relu', strides=(1, 1)))
    net.add(MaxPooling2D(pool_size = (2,2)))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    
    #Flatten layer
    net.add(Flatten())
    
    #Dense Layer 0
#    net.add(Dense(512, activation='relu'))
#    net.add(BatchNormalization())
#    net.add(Dropout(dropout))
    
    #Dense Layer 1
    net.add(Dense(256, activation='relu'))
    net.add(BatchNormalization())
    net.add(Dropout(dropout))
    
    #Dense Layer 2
#    net.add(Dense(128, activation='relu'))
#    net.add(BatchNormalization())
#    net.add(Dropout(dropout))
    
    #Dense Layer 3
#    net.add(Dense(64, activation='relu'))
#    net.add(BatchNormalization())
#    net.add(Dropout(dropout))
    
    #Dense Layer 4
#    net.add(Dense(32, activation='relu'))
#    net.add(BatchNormalization())
#    net.add(Dropout(dropout))
    
    #Dense Layer 5
#    net.add(Dense(16, activation='relu'))
#    net.add(BatchNormalization())
#    net.add(Dropout(dropout))
    
    net.add(Dense(units = outputshape, activation='softmax'))
    
    net.summary()
    
    return net