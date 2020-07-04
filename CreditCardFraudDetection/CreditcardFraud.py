# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:55:12 2020

@author: mnsah
"""

import pandas as pd
import numpy as np



file = 'D:/Data/Machine_Learning/CreditCardFraud/creditcard.csv' 
data = pd.read_csv(file)
#print(data.head)
#print(data.describe())
pos_fraction = len(data[data['Class']==1])*100/len(data)
neg_fraction = len(data[data['Class']==0])*100/len(data)
print([neg_fraction, pos_fraction])