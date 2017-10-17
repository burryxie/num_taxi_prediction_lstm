# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:35:28 2017

@author: Xie-Kaiqiang
"""
import os
path = r'C:\Users\Xie\Documents\GitHub\num_taxi_prediction_lstm'
os.chdir(path)

import pandas as pd
import numpy as np
from series_to_supervised import series_to_supervised



def load_data(filename):
    
    data = pd.read_csv(filename,sep=';')
    data = data.iloc[:,0:624]
    data = data.values
    indexs = list(np.where(data.sum(axis=1) > data.sum(axis=1).mean()))
    data = data[indexs,:].reshape([len(indexs[0]),data.shape[1]])
    pd.DataFrame(data).to_csv('Useful_TAZ.csv',index=False)
    #del indexs
    data = data.transpose()
    return data


data = load_data('TAZ.csv')
origin,destination = data[0:312,:],data[312:,:]

o_train_x,o_train_y,o_test_x,o_test_y = series_to_supervised(origin,n_in=3,perc_train=0.7)
d_train_x,d_train_y,d_test_x,d_test_y = series_to_supervised(destination,n_in=3,perc_train=0.7)

# reshape input to be [samples, time steps, features]
o_train_x = o_train_x.reshape()