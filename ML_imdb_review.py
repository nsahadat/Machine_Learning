# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:39:22 2019

@author: mnsah
"""

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

font = {'font.family' : 'normal',
        'font.weight' : 'normal',
        'font.size'   : 14}

plt.rcParams.update(font)


from keras.datasets import imdb
(train_data, train_lab), (test_data, test_lab) = imdb.load_data(num_words = 10000)


def vectorize_sequence(sequence, dimension = 10000):
    result = np.zeros((len(sequence),dimension))
    
    for i, sequence in enumerate(sequence):
        result[i, sequence] = 1.
            
    return result

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence (test_data)

#print(train_data[0])
#print(x_train.shape)

y_train = np.asarray(train_lab).astype('float32')
y_test = np.asarray(test_lab).astype('float32')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation = tf.keras.activations.relu, input_shape = (10000, )))
model.add(tf.keras.layers.Dense(16, activation = tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid))

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = 0.001), 
              loss = tf.keras.losses.binary_crossentropy, 
              metrics = [tf.keras.metrics.binary_accuracy])

# divide values to train and validation sets
x_val = x_train[:10000]
x_train_part = x_train[10000:]

y_val = y_train[:10000]
y_train_part = y_train[10000:]



trainRes = model.fit(x_train_part, 
                     y_train_part, 
                     epochs = 20, 
                     batch_size = 512,
                     validation_data = (x_val, y_val))

#%% plotting part
#print(trainRes.history.keys())


train_loss = trainRes.history['loss']
val_loss = trainRes.history['val_loss']

train_acc = trainRes.history['binary_accuracy']
val_acc = trainRes.history['val_binary_accuracy']

ep = range(1, (len(train_loss) +1) )

plt.figure(figsize=(3,2), dpi=320)
plt.plot(ep, train_loss, 'bo', label = 'train loss', linewidth = 2)
plt.plot(ep, val_loss, 'b', label = 'val loss', linewidth = 2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.title('Loss vs. Epoch graph')


plt.figure(figsize=(3,2), dpi=320)
plt.plot(ep, train_acc, 'ro', label = 'train acc', linewidth = 2)
plt.plot(ep, val_acc, 'r', label = 'val acc', linewidth = 2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.title('Accuracy vs. Epoch graph')

#%% finally change the model file epochs to evaluate with testing data
model.fit(x_train, 
          y_train, 
          epochs = 4, 
          batch_size = 512)

(loss, acc) = model.evaluate(x_test, y_test)
print ('loss : ', loss)
print('Accuracy(%) :', acc*100)







 