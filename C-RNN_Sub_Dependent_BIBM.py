
from __future__ import print_function
import numpy as np

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
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import pickle

# size1 channel number size2 scale number
size1 = 32
size2 = 32

subNum =32
window = 1
trialTime = 60
maxToAdd = trialTime/window

batch_size = 50
nb_epochs = 50

# 0 valence 1 arousal
emodim = 1

# add noise to augment
exNum = 250


#def td_avg(x):
    #return K.mean(x)
# model.add(TimeDistributed(Dense(2, activation='softmax')))
# model.add(Lambda(td_avg))

time_distributed_merge_layer = Lambda(function=lambda x: K.mean(x, axis=1),
                                      output_shape=lambda shape: (shape[0],) + shape[2:])

for subNo in range(1,32):
    file = h5py.File('D:\\LX\\CRNNXY\\trial_cwt\\nooverlap\\'+str(window)+'s_as_element\\sub'+str(subNo+1)+'.mat', 'r')
    X = file['X']
    Y = file['Y_personal']
    Y40 = file['Y_40personal']
    X = np.transpose(X)
    Y = np.transpose(Y)[:, emodim]
    Y40 = np.transpose(Y40)[:, emodim]
    X = X[:,:,:,:,6:38] #inut frame, scale selection
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
        model = Sequential()
        # input shape variates with different frame shapes, 2D or 3D
        model.add(TimeDistributed(Convolution2D(8, 32, 1, border_mode='valid'), input_shape=(maxToAdd, 1, size1, size2)))
        model.add(Activation('relu'))
        model.add(TimeDistributed(AveragePooling2D(pool_size=(1, 2), border_mode='valid')))
        model.add(TimeDistributed(Convolution2D(16, 1, 1, border_mode='valid')))
        model.add(Activation('relu'))
        model.add(TimeDistributed(AveragePooling2D(pool_size=(1, 2), border_mode='valid')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(output_dim=128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(256)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(2, activation='softmax')))
        model.add(time_distributed_merge_layer)

        rmsprop = RMSprop()
        sgd = SGD(momentum=0.9)
        adadelta = Adadelta()
        nadam = Nadam()
        adagrad = Adagrad()

        model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
        model.summary()
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs, validation_data=(X_test, Y_test), verbose=1)

        foldNo=foldNo+1

        del X_train
        del X_test
        del Y_train
        del Y_test

