#! /usr/bin/env python2.7
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

''' This script generates a submission to Otto competition from an ensemble of 
Deep Neural Networks and Gradient Boosting Machines. 
All predictions are averaged in a single output
object and saved to submit.csv.

Luis Andre Dutra e Silva - http://kaggle.com/luisandre
'''

#generate labels for DNN and GBM
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        yt = np_utils.to_categorical(y)
    return y, yt

#load csv data into numpy arrays
def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids
   
print("Loading data...")

#scaling data increases LV precision
min_max_scaler = preprocessing.StandardScaler()
X, labels = load_data('train.csv', train=True)
y, yt = preprocess_labels(labels)
T, ids = load_data('test.csv', train=False)
X = min_max_scaler.fit_transform(X)
T = min_max_scaler.fit_transform(T)
xg_train = xgb.DMatrix(X, label=y)
xg_test = xgb.DMatrix(T)
p_w = np.zeros((T.shape[0],9),np.float32)

#generate an ensemble of 10 DNN models and 10 GBM models 
for r in range(10):
    nb_classes = yt.shape[1]
    print(nb_classes, 'classes')

    dims = X.shape[1]
    print(dims, 'dims')
    
    np.random.seed((r+1)*10)
    
    model = Sequential()
    model.add(Dense(dims, 1000, init='glorot_uniform'))
    model.add(PReLU((1000,)))
    model.add(BatchNormalization((1000,)))
    model.add(Dropout(0.5))
    
    model.add(Dense(1000, 500, init='glorot_uniform'))
    model.add(PReLU((500,)))
    model.add(BatchNormalization((500,)))
    model.add(Dropout(0.5))
    
    
    model.add(Dense(500, nb_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer="adagrad")
    
    print("Training DNN model...")
    
    model.fit(X, yt, nb_epoch=150, batch_size=32, validation_split=0.1)
    
    
    p_w = p_w + model.predict_proba(T)
    
    print("Cross validating GBM model...")    

    param = {"objective":"multi:softprob",
               "eval_metric":"mlogloss",
               "num_class":9,
               "eta":0.02,
               "gamma":0.0,
               "max_depth":10,
               "max_delta_step":0,
               "subsample":0.6,
               "colsample_bytree":0.9,
               "base_score":0.5,
               "seed":(r+1)*10,
               "nthread":8}
    
    nround = 1200
    history = xgb.cv(param, xg_train, nround,nfold=3)
    scores = np.zeros((nround,1),np.float32)
    i = 0
    for line in history:
      exp1 = line.split('\t')
      exp2 = exp1[1].split(':')
      exp3 = exp2[1].split('+')
      scores[i] = np.float32(exp3[0])
      i = i+1
    print(scores.argmin()+1,scores.min())

    nround = scores.argmin()+1

    print("Training GBM model...")    
    
    bst = xgb.train(param, xg_train,nround)
    
    p_w = p_w + bst.predict( xg_test )
  

with open('submit.csv', 'w') as submit_file:
  submit_file.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
  for i in range(len(p_w)):
    submit_file.write(`i+1`+',')
    for j in range(p_w.shape[1]):
      submit_file.write(("%.8f"%p_w[i,j]))
      if j == p_w.shape[1]-1:
        submit_file.write('\n')
      else:
        submit_file.write(',')
