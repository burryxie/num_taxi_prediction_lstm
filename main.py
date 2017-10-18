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
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from series_to_supervised import series_to_supervised
from matplotlib import pyplot



def load_data(filename):
    data = pd.read_csv(filename,sep=';')
    data = data.iloc[:,0:624]
    data = data.values
    indexs = list(np.where(data.sum(axis=1) > data.sum(axis=1).mean()))
    data = data[indexs,:].reshape([len(indexs[0]),data.shape[1]])
    #pd.DataFrame(data).to_csv('Useful_TAZ.csv',index=False)
    #del indexs
    data = data.transpose()
    return data


data = load_data('TAZ.csv')
origin,destination = data[0:312,:],data[312:,:]

#convert the dataframe to the supervised format 
reframed_origin = series_to_supervised(origin,n_in=3)
reframed_origin = reframed_origin.values
reframed_destination = series_to_supervised(destination,n_in=3)
reframed_destination = reframed_destination.values

#split into trainning and test sets
perc_train = 0.7
num_train_origin = round(len(reframed_origin)*perc_train)
num_train_destination = round(len(reframed_destination)*perc_train)

train_origin = reframed_origin[:num_train_origin,:]
test_origin  = reframed_origin[num_train_origin:,:]

train_destination = reframed_destination[0:num_train_destination,:]
test_destination  = reframed_destination[num_train_destination:,:]

#split into x and y
train_origin_x,train_origin_y = train_origin[:,:1476],train_origin[:,1476:]
test_origin_x,test_origin_y = train_origin[:,:1476],train_origin[:,1476:]

train_destination_x,train_destination_y = train_destination[:,:1476],train_destination[:,1476:]
test_destination_x,test_destination_y = test_destination[:,:1476],test_destination[:,1476:]

del origin,destination,num_train_origin,num_train_destination,perc_train
del train_origin,test_origin,train_destination,test_destination
del reframed_origin,reframed_destination


# reshape input to be [samples, time steps, features],
# where samples is the number of samples, 
# time steps is the number of steps used to predict next time step
# and features is the number of features for one sample
train_origin_x = train_origin_x.reshape((train_origin_x.shape[0],3,492))
train_destination_x = train_destination_x.reshape((train_destination_x.shape[0],3,492))

# modelling
hidden_units = 50
num_epochs = 10
num_batchs = 8

model = Sequential()
model.add(LSTM(hidden_units,return_sequences=True,
                   input_shape=(train_origin_x.shape[1],train_origin_x.shape[2])))
model.add(Dense(492))
model.compile(loss='mse',optimizer='adam')
print('model trainning begins...')
history = model.fit(train_origin_x,train_origin_y,epochs = num_epochs,batch_size = num_batchs,
              validation_data=(test_origin_x,test_origin_y),verbose=2,shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
yhat = model.predict(test_origin_x)
test_x = test_origin_x.reshape((test_origin_x.shape[0], test_origin_x.shape[2]))

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
