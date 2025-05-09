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
import xgboost as xgb
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

#xgboost parameters
param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax', 'eval_metric': 'auc'}
train = 0

trainData = pd.read_csv('D:/Data/Machine_Learning/digit-recognizer/train.csv')
testData = pd.read_csv('D:/Data/Machine_Learning/digit-recognizer/test.csv')

options = int(input('1. show some digits\n2. train with SVM\n3. Train with LeNet5\n4. Train with custom NN\n5. Train with boosting algorithm\n6. Train SVM with Grid Search\n'))

# show the digits how it looks like
if options==1:
    plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        digit= trainData.iloc[i,1:].to_numpy().reshape(28,28)
        plt.imshow(digit)

# process the input outputs
x, y = trainData.iloc[:, 1:].to_numpy(), trainData.iloc[:, 0].to_numpy()
x = np.round(x/255) #rgb to gray
x_test = testData.iloc[:,:].to_numpy()
x_test = np.round(x_test/255)

x_tr, x_val, y_tr, y_val = train_test_split(x,y, test_size = 0.1, random_state = 33)

# for deep learning training
if options==3 or options==4:
    x_tr = x_tr.reshape(x_tr.shape[0], 28, 28, 1)
    x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x = x.reshape(x.shape[0], 28, 28, 1)
    
    y_tr = keras.utils.to_categorical(y_tr, num_classes = 10)
    y = keras.utils.to_categorical(y, num_classes = 10)
    y_v = y_val
    y_val = keras.utils.to_categorical(y_val, num_classes = 10)

if options==3:
    model = LeNetMod5((x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]), y_tr.shape[1])
if options==4:
    model = simple2DModel02((x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]), y_tr.shape[1])
    
if options==3 or options==4:  
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# early stopping part
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.0001) 
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# compile the model
    history=model.fit(x_tr, y_tr ,epochs=50, callbacks=[mc], batch_size=32, validation_data=(x_val,y_val))
#    history=model.fit(x, y, epochs=100, batch_size=32)

# visualize the learning
    plt.figure()
    plt.plot(history.history['loss'], label='train') 
    plt.plot(history.history['val_loss'], label='val') 
    plt.ylabel('loss')
    plt.grid()
    plt.legend() 

if options==2:
#    clf = SVC(0.1)
    clf = SVC(gamma = 0.001)
    clf.fit(x_tr, y_tr)
    y_val_est = clf.predict(x_val)

if options==5:
    clf = xgb.XGBClassifier(max_depth=4, objective='multi:softmax', n_estimators=100, 
                        num_classes=10, eval_metric = 'auc', learning_rate = 0.5)
    clf.fit(x_tr, y_tr)
    y_val_est = clf.predict(x_val)
    y_est = clf.predict(x_test)
    
    

# grid search to train the best classifier
if options==6:
    clf = GridSearchCV(SVC(), parameters, cv = 5, scoring = 'roc_auc')
    if train==1:
        clf = GridSearchCV(SVC(), parameters, cv = 5)
        clf.fit(x, y)
        dump(clf, 'D:/GitHub/Machine_Learning/HandWrittenLetter/svm_optimized.joblib')
    else:
        clf = load('D:/GitHub/Machine_Learning/HandWrittenLetter/svm_optimized.joblib')
        
    y_val_est = clf.predict(x_val)

#for deep learning
if options==3 or options==4:
    model = load_model('best_model.h5')
    prob=model.predict(x_val)
    y_val_est = np.argmax(prob, axis = 1)
    
    plot_confusion_matrix(y_val_est, y_v, target_names = [0,1,2,3,4,5,6,7,8,9])
    
    prob_test = model.predict(x_test)
    y_est = np.argmax(prob_test, axis = 1)

if options==2 or options==5 or options==6:
    plot_confusion_matrix(y_val_est, y_val, target_names = [0,1,2,3,4,5,6,7,8,9])

if options==2 or options==6:
    y_est = clf.predict(x_test)

# find the results and save it in the csv
if options!=1:
    result = pd.DataFrame()
    result['ImageId'] = range(1,28001)
    result['Label'] = y_est
    result.to_csv('output.csv',index = False)

