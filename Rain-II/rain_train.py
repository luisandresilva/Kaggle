# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:51:05 2015

@author: luisandresilva
"""

import numpy as np
import math
import random

np.random.seed(1969)

from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

max_len = 30
dims = 24
max_count = 1200000
def get_samples(filename):
    source = np.zeros((max_count,max_len,dims),np.float32)
    target = np.zeros((max_count,1),np.float32)
    count = 0
    with open(filename,"r") as source_file:
        source_file.readline()
        src = source_file.readline().replace("\n","")
        cols = src.split(",")
        while count < max_count and src != "":
            lastid = cols[0]
            rainfall = float(cols[len(cols)-1])
            allNull = True
            srcs = []
            while cols[0] == lastid:
                srcs.append(src)
                if cols[3].strip() != '':
                    allNull = False
                src = source_file.readline().replace("\n","")
                cols = src.split(",")                               
            if allNull:
                continue
            print count
            target[count] = rainfall
            seq = np.zeros((max_len,dims),np.float32)
            length = 0
            for s in srcs:
                c = s.split(",")
                if len(c) > 1:
                    minute = float(c[1])
                    seq[length,0] = minute
                    for i in range(1,dims-2):
                        if c[i+1].strip() =="":
                            val = 0
                        else:
                            val = float(c[i+1])
                        if i in [2,3,4,5,6,7,8,9] and np.isfinite(val):
                            val = math.pow(math.pow(10,val/10)/200,0.625)
                        if i in [14,15,16,17]:
                            val = math.pow(10,val/10)
                        seq[length,i] = val
                    seq[length,22] = float(len(srcs))
                    seq[length,23] = np.mean(seq[length,0:22])
                    length += 1
                    source[count,:,:] = seq
            count += 1
    X = source[0:count,:,:]
    Y = target[0:count,:]
    return X,Y


X_train,Y_train = get_samples('../input/train.csv')
np.random.seed(9333)
model = Sequential()
model.add(LSTM(dims,35,init="uniform",inner_init="glorot_uniform",truncate_gradient=-1,activation="sigmoid"))
model.add(Dense(35,1,activation="linear"))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

kf = KFold(len(X_train), n_folds=50)
seed = 1
hidden = []
dropouts = []
for train, test in kf:
    checkpointer = ModelCheckpoint(filepath="net/lstm"+`seed`+".hdf5", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    X = X_train[train]
    Y = Y_train[train]
    X_t = X_train[test]
    Y_t = Y_train[test]
    history = model.fit(X, Y, nb_epoch=100, batch_size=256, verbose=1, validation_data=[X_t,Y_t], callbacks=[checkpointer, earlystopper])  
    seed += 1    



        