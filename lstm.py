# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:50:09 2017

@author: Xie
"""

from keras.models import Sequential
from keras.layers.core import Activation,Dense,Dropout
from keras.layers.model import LSTM

del model(hidden_units,in_out_nuerons):
    model = Sequential()
    model.add(LSTM(hidden_units,return_sequences=True,
                   input_size=(None,)))
    
