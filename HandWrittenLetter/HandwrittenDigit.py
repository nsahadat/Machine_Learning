# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:07:48 2020

@author: mnsah
"""

import numpy as np
import os
import keras
import pandas as pd
import PIL as image
import cv2 as cv
from joblib import dump, load
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from DeepModels import LeNetMod5, simple2DModel02
#from sklearn.metrics import plot_confusion_matrix
from plotCF import plot_confusion_matrix


plt.rcParams.update({'font.size':20, 'font.weight':'bold'})

# parameter search
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma':[0.0001, 0.001, 0.01, 0.1]}
train = 0

trainData = pd.read_csv('D:/Data/Machine_Learning/digit-recognizer/train.csv')
testData = pd.read_csv('D:/Data/Machine_Learning/digit-recognizer/test.csv')

# show the digits how it looks like
#for i in range(10):
#    digit = trainData.iloc[i, 1:].to_numpy().reshape(28,28)
#    plt.figure(i)
#    plt.imshow(digit)

# simple support vector classifier
x, y = trainData.iloc[:, 1:].to_numpy(), trainData.iloc[:, 0].to_numpy()
x = np.round(x/255) #rgb to gray
x_test = testData.iloc[:,:].to_numpy()
x_test = np.round(x_test/255)

x_tr, x_val, y_tr, y_val = train_test_split(x,y, test_size = 0.01, random_state = 33)

# for deep learning training
x_tr = x_tr.reshape(x_tr.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x = x.reshape(x.shape[0], 28, 28, 1)

y_tr = keras.utils.to_categorical(y_tr, num_classes = 10)
y = keras.utils.to_categorical(y, num_classes = 10)
y_v = y_val
y_val = keras.utils.to_categorical(y_val, num_classes = 10)

model = LeNetMod5((x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]), y_tr.shape[1])
#model = simple2DModel02((x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]), y_tr.shape[1])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# early stopping part
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# compile the model
history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

# visualize the learning
plt.figure()
plt.plot(history.history['loss'], label='train') 
plt.plot(history.history['val_loss'], label='val') 
plt.ylabel('loss')
plt.grid()
plt.legend() 

#clf = SVC(0.1)
#clf = SVC(gamma = 0.001)
#clf.fit(x_tr, y_tr)
#y_est = clf.predict(x_val)

#plot_confusion_matrix(y_est, y_val, target_names = [0,1,2,3,4,5,6,7,8,9])

# grid search to train the best classifier
#clf = GridSearchCV(SVC(), parameters, cv = 5, scoring = 'roc_auc')
#if train==1:
#    clf = GridSearchCV(SVC(), parameters, cv = 5)
#    clf.fit(x, y)
#    dump(clf, 'D:/GitHub/Machine_Learning/HandWrittenLetter/svm_optimized.joblib')
#else:
#    clf = load('D:/GitHub/Machine_Learning/HandWrittenLetter/svm_optimized.joblib')
#print(sorted(clf.cv_results_.keys()))
#print(clf.cv_results_['param_C'])
#y_val_est = clf.predict(x_val)

#for deep learning
model = load_model('best_model.h5')
prob=model.predict(x_val)
y_val_est = [np.argmax(prob[k]) for k in range(len(prob))]
plot_confusion_matrix(y_val_est, y_v, target_names = [0,1,2,3,4,5,6,7,8,9])

#y_est = clf.predict(x_test)

# for deep learning
prob_test = model.predict(x_test)
y_est = [np.argmax(prob_test[k]) for k in range(len(prob_test))]

# find the results and save it in the csv
result = pd.DataFrame()
result['ImageId'] = range(1,28001)
result['Label'] = y_est
result.to_csv('output.csv',index = False)

