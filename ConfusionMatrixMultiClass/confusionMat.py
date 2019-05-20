# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:39:24 2019

@author: MdNazmus
"""
import csv
import numpy as np
data_path ='outcome.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    #headers = next(reader)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data)

#print(data.shape)
        
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data[:,0], data[:,1])
print(cm)
FP = (cm.sum(axis=0) - np.diag(cm))
FN = (cm.sum(axis=1) - np.diag(cm)) 
TP = (np.diag(cm))
TN = cm.sum() - (FP + FN + TP)

accr = (TP + TP)/(TP + TP + FP + FN)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)

print(accr.mean()) 
print(TPR.mean())
print(TNR.mean())
