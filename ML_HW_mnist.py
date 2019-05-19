# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:04:27 2019

# Inspired by deep learning with Python
@author: mnsah
"""

# for the dataset loading
from keras.datasets import mnist 
#for building network
from keras import models
from keras import layers
# to convert the data format 
from keras.utils import to_categorical


# load the dataset to the following variables with labels
(train_data, train_lab), (test_data, test_lab) = mnist.load_data()

print('Loading dataset succesful.........')

# let's build the network
# create type of network
network = models.Sequential()
# add hidden layers
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
# add output layer
network.add(layers.Dense(10, activation = 'softmax'))

# let's define optimizer, loss, metric
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# converting dataset to the right format
train_data = train_data.reshape(len(train_data),28*28)
#normalize the values and converting to floating point
train_data = train_data.astype('float32')/255
# converting dataset to the right format
test_data = test_data.reshape(len(test_data),28*28)
#normalize the values and converting to floating point
test_data = test_data.astype('float32')/255

# converting the label formating
train_lab = to_categorical(train_lab)
test_lab = to_categorical(test_lab)

# train the data
network.fit(train_data, train_lab, epochs = 5, batch_size = 128)

# test the performance
test_loss, test_acc = network.evaluate(test_data, test_lab)

# print the outputs
print('test loss = ', test_loss)
print('Accuracy (%) = ', test_acc*100)





