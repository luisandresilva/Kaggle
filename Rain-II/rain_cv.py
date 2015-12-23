# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:51:05 2015

@author: luisandresilva
"""

import numpy as np
import math
np.random.seed(1969)

from keras.models import Sequential
from keras.layers.core import Dropout, Dense
from keras.layers.recurrent import LSTM
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
import cPickle

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
L = np.zeros((50,int(X_train.shape[0]/50)),np.float32)
MAE = 0.0
for train, test in kf:
    X = X_train[train]
    Y = Y_train[train]
    X_t = X_train[test]
    Y_t = Y_train[test]
    print seed
    model.load_weights('net/lstm'+`seed`+'.hdf5')
    scores = model.predict(X_t,batch_size=256,verbose=1)
    MAE += mean_absolute_error(Y_t, scores, sample_weight=None)
    for i in range(L.shape[1]):
        L[seed-1,i] = (scores[i,0]-Y_t[i,0])**2
    seed += 1    
L_HAT = 0
MAE /= 50
for k in range(L.shape[0]):
    for i in range(L.shape[1]):
        L_HAT += L[k,i]
L_HAT /= float(L.shape[0]*L.shape[1])
EPE = L_HAT
n = L.shape[1]
K = L.shape[0]
THETA2 = 0
for k in range(L.shape[0]):
    for i in range(L.shape[1]):
        THETA2 += (L[k,i]-L_HAT)**2
THETA2 = float(1/float(K*n*(K*n-1)))*THETA2;
with open('../output/cv/cv.csv','a') as cv_file:
    cv_file.write('50-fold 1 LSTM 35 1 DENSE normal with outliers rmsprop 24 dims MAE 256,'+ `EPE`+','+`THETA2`+','+`MAE`+'\n')




        