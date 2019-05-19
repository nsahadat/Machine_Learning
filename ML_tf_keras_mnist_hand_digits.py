# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:41:27 2019

@author: mnsah
"""
# using tensor flow with keras
import tensorflow as tf
from tensorflow import keras

# for the dataset loading
from keras.datasets import mnist 


# load the dataset to the following variables with labels
(train_data, train_lab), (test_data, test_lab) = mnist.load_data()

print('Loading dataset succesful.........')

# let's build the network
# create type of network
model = tf.keras.Sequential([keras.layers.Dense(512, activation = tf.nn.relu, input_shape = (28*28, )), 
                        keras.layers.Dense(10, activation = tf.nn.softmax)
                        ])

# let's define optimizer, loss, metric
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# converting dataset to the right format
train_data = train_data.reshape(len(train_data),28*28)
#normalize the values and converting to floating point
train_data = train_data.astype('float32')/255
# converting dataset to the right format
test_data = test_data.reshape(len(test_data),28*28)
#normalize the values and converting to floating point
test_data = test_data.astype('float32')/255

# converting the label formating
train_lab = tf.keras.utils.to_categorical(train_lab)
test_lab = tf.keras.utils.to_categorical(test_lab)

# train the data
model.fit(train_data, train_lab, epochs = 5, batch_size = 128)

# test the performance
test_loss, test_acc = model.evaluate(test_data, test_lab)

# print the outputs
print('test loss = ', test_loss)
print('Accuracy (%) = ', test_acc*100)