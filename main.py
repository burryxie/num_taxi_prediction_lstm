# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:35:28 2017

@author: Xie-Kaiqiang
"""
import os
path = r'C:\Users\Xie-Kaiqiang\Documents\GitHub\num_taxi_prediction_lstm'
os.chdir(path)

import pandas as pd
import numpy as np
import series_to_supervised



def load_data():
    
    data = pd.read_csv('TAZ.csv',sep=';')
    data = data.iloc[:,0:624]
    data = data.values
    indexs = list(np.where(data.sum(axis=1) > data.sum(axis=1).mean()))
    data = data[indexs,:].reshape([len(indexs[0]),data.shape[1]])
    #del indexs
    data = data.transpose()
    return data


data = load_data()
data = series_to_supervised(data,n_in=5,n_out=1)