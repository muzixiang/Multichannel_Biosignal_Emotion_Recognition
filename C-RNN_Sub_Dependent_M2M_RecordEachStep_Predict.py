from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential

from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD, Adadelta, Nadam, Adagrad
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda
from keras.regularizers import l2

from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
import h5py
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as cp
from keras.models import model_from_json

# define some run parameters
size1 = 32
size2 = 32

subNum =32
window = 1
trialTime = 60
maxToAdd = trialTime/window

# 0 valence 1 arousal
emodim = 0

# add noise to augment
exNum = 250

sqs = []

batch_size = 50
hiddendimension = 2

def td_avg(x):
    return K.mean(x, axis=1)
def td_avg_shape(x):
    return tuple((batch_size, hiddendimension))

#output_shape=lambda shape: (shape[0],) + shape[2:]
time_distributed_merge_layer = Lambda(function=td_avg, output_shape=td_avg_shape)

subNo = 0
# your directory that stores the 2D frames
subfile = h5py.File('D:\\LX\\RCNNXY\\trial_cwt\\nooverlap\\'+str(window)+'s_as_element\\sub'+str(subNo+1)+'.mat', 'r')
X = subfile['X']
Y = subfile['Y_personal']
Y40 = subfile['Y_40personal']
X = np.transpose(X)
Y = np.transpose(Y)[:, emodim]
Y40 = np.transpose(Y40)[:, emodim]
X = X[:,:,:,:,6:38]
print('X shape:', X.shape)
print('Y shape:', Y.shape)
print('Y40 shape:', Y40.shape)
print('Y40 label: %s' % Y40)
print("Building model for ------------------------------------------------------------------------------------------------ sub "+str(subNo+1))


# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2016)
kfolds = StratifiedKFold(Y40, n_folds=5, shuffle=False, random_state=2016)
foldNo=1
for train_indices, test_indices in kfolds:
    print('kfold cross validation fold starting ------------------------------------------------------------------------------fold '+str(foldNo))
    print('Train: %s | Test: %s' % (train_indices, test_indices))
    ex_train_indices = []
    for train_indice in train_indices:
        start_indice = train_indice*exNum
        for exNo in range(0, exNum):
            ex_train_indices.append(start_indice + exNo)
    ex_test_indices = []
    for test_indice in test_indices:
        start_indice = test_indice*exNum
        for exNo in range(0, exNum):
            ex_test_indices.append(start_indice + exNo)
    # print('Expand Train: %s' % (ex_train_indices))
    # print('Expand Test: %s' % (ex_test_indices))

    X_train = X[ex_train_indices]
    Y_train = Y[ex_train_indices]
    X_test = X[ex_test_indices]
    Y_test = Y[ex_test_indices]

    Y_train = np_utils.to_categorical(Y_train, 2)
    Y_test = np_utils.to_categorical(Y_test, 2)
    print('Y shape:', Y_train.shape)

    # copy Y to  sequence length
    Y_train_seq = []
    for i in range(len(Y_train)):
        ext_ele = []
        for j in range(maxToAdd):
            ext_ele.append(Y_train[i])
        Y_train_seq.append(ext_ele)

    Y_test_seq = []
    for i in range(len(Y_test)):
        ext_ele = []
        for j in range(maxToAdd):
            ext_ele.append(Y_test[i])
        Y_test_seq.append(ext_ele)

    model = Sequential()
    model.add(TimeDistributed(Convolution2D(8, 32, 1, border_mode='valid'), input_shape=(maxToAdd, 1, size1, size2)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling2D(pool_size=(1, 2), border_mode='valid')))
    model.add(TimeDistributed(Convolution2D(16, 1, 1, border_mode='valid')))
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling2D(pool_size=(1, 2), border_mode='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(output_dim=128, return_sequences=True))
    #model.add(LSTM(output_dim=64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256)))
    model.add(Dropout(0.5))
    tdlayer = TimeDistributed(Dense(2, activation='softmax'))
    model.add(tdlayer)
    model.add(time_distributed_merge_layer)

    model.load_weights('M2M_model_weights_sub'+str(subNo+1)+'_fold'+str(foldNo)+'.h5')

    get_td_layer_output = K.function([model.layers[0].input, K.learning_phase()], tdlayer.output)

    sel_index = range(0,7750,exNum)
    layer_output = get_td_layer_output([X_train[sel_index,:,:,:,:], 0])

    print('td layer output shape:', layer_output.shape)

    sio.savemat('predict_each_step_sub'+str(subNo+1)+'_fold'+str(foldNo)+'.mat', {'online_predictions':layer_output})

    foldNo = foldNo + 1
