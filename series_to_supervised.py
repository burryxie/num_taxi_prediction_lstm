# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:16:37 2017

@author: Xie-Kaiqiang
"""
from pandas import DataFrame,concat

# convert series to supervised learning
def series_to_supervised(data, n_in=1, perc_train=0.7):
    x,y = [],[]
    for i in range(data.shape[0]-n_in):
        x.append(data[i:i+n_int,:].flatten())
        y.append(data[i+n_int,:].flatten())
    
    num_train = round(data.shape[0]*perc_train)
    train_x,train_y = x[0:num_train],y[0:num_train]
    test_x,test_y = x[num_train:],y[num_train:]
    
	return train_x,train_y,test_x,test_y