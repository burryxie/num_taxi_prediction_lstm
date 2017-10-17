# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:16:37 2017

@author: Xie-Kaiqiang
"""
import numpy as np

# convert series to supervised learning
def series_to_supervised(data, n_in=1,n_out=1, perc_train=0.7):
    x,y = [],[]
    for i in range(data.shape[0]-n_in-1):
        x.append(data[i:i+n_in,:].flatten())
        y.append(data[i+n_in,:].flatten())
    
    num_train = round(data.shape[0]*perc_train)
    train_x,train_y = x[0:num_train],y[0:num_train]
    test_x,test_y = x[num_train:],y[num_train:]
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    return train_x,train_y,test_x,test_y