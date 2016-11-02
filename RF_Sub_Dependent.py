from __future__ import division
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio
from sklearn.cross_validation import StratifiedKFold
import numpy as np

subNum = 32
trialNum = 40

file1 = sio.loadmat(file_name='D:\\eeg_handfeatures.mat')
file2 = sio.loadmat(file_name='D:\\trial_labels_personal_valence_arousal_dominance.mat')
X = file1['eeg_handfeatures']
Y = file2['trial_labels']
print('Y shape:', Y.shape)
# 0 valence 1 arousal
emodim = 0

for subNo in range(0, subNum):
    print("Building model for ------------------------------------------------------------------------------------------------ sub " + str(subNo + 1))
    subX = X[subNo*trialNum:(subNo+1)*trialNum]
    subY = Y[subNo*trialNum:(subNo+1)*trialNum,emodim]
    kfolds = StratifiedKFold(subY, n_folds=5, shuffle=False, random_state=2016)
    foldNo = 1
    accs = []
    for train_indices, test_indices in kfolds:
        print('kfold cross validation fold starting ------------------------------------------------------------------------------fold ' + str(foldNo))
        print('Train: %s | Test: %s' % (train_indices, test_indices))
        subX_train = subX[train_indices]
        subX_test = subX[test_indices]
        subY_train = subY[train_indices]
        subY_test = subY[test_indices]
        rf = RandomForestClassifier(criterion="gini", n_estimators=10, random_state=2016)
        rf.fit(subX_train,subY_train)
        predict_subY = rf.predict(subX_test)
        print('predict_subY shape:', predict_subY.shape)
        rightNum = 0
        testNum = len(test_indices)
        print('test Num: %d'%(testNum))
        for testNo in range(0,testNum):
            if subY_test[testNo] == predict_subY[testNo]:
                rightNum = rightNum+1
        print('predict right num: %d'%(rightNum))
        acc = rightNum/testNum*100
        accs.append(acc)
        print('acc is: %.5f'%(acc))
        foldNo = foldNo + 1
    avgacc = np.mean(accs)
    print('================================average acc is: %.5f'%(avgacc))