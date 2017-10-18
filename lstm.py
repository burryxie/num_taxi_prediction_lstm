# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:50:09 2017

@author: Xie
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.model import LSTM

def model(hidden_units,train_x,train_y,test_x,test_y,num_epochs,num_batchs):
    model = Sequential()
    model.add(LSTM(hidden_units,return_sequences=True,
                   input_size=(train_x.shape[1],train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    print('model trainning begins...')
    model.fit(train_x,train_y,epochs = num_epochs,batch_size = num_batchs,
              validation_data=(test_x,test_y),verbose=2,shuffle=False)
    
    
