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
#from blstm import BiDirectionLSTM
from keras.layers.recurrent import LSTM
import cPickle

max_len = 30
dims = 24
max_count = 2000000
def get_samples(filename):
    source = np.zeros((max_count,max_len,dims),np.float32)
    target = np.zeros((max_count,1),np.int32)
    count = 0
    with open(filename,"r") as source_file:
        source_file.readline()
        src = source_file.readline().replace("\n","")
        cols = src.split(",")
        while count < max_count and src != "":
            lastid = cols[0]
            srcs = []
            while cols[0] == lastid:
                srcs.append(src)
                src = source_file.readline().replace("\n","")
                cols = src.split(",")
            print count
            target[count] = int(lastid)
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


X_test,Id = get_samples('../input/test.csv')
np.random.seed(9333)
model = Sequential()
model.add(LSTM(dims,35,init="uniform",inner_init="glorot_uniform",truncate_gradient=-1,activation="sigmoid"))
model.add(Dense(35,1,activation="linear"))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

preds = np.zeros((Id.shape[0],1),np.float32)
for seed in range(1,51):
    print seed
    model.load_weights('net/lstm'+`seed`+'.hdf5')
    preds = preds + model.predict(X_test,batch_size=256,verbose=1)

preds /= 50

with open('../output/submit_lstm.csv','w') as submit_file:
    submit_file.write('Id,Expected\n')
    for i in range(preds.shape[0]):
        submit_file.write(str(Id[i,0])+","+str(preds[i,0])+"\n")

        