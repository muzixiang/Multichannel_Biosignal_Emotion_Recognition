from __future__ import division
from sklearn import svm
import scipy.io as sio
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
import numpy as np

subNum = 32
trialNum = 40

file1 = sio.loadmat(file_name='.//eeg_handfeatures.mat')
file2 = sio.loadmat(file_name='.//trial_labels_personal_valence_arousal_dominance.mat')
X = file1['eeg_handfeatures']
Y = file2['trial_labels']
print('Y shape:', Y.shape)
# 0 valence 1 arousal
emodim = 1

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

        min_max_scaler = preprocessing.MinMaxScaler()
        subX_train_scale = min_max_scaler.fit_transform(subX_train)

        classifier = svm.SVC()
        classifier.fit(subX_train_scale,subY_train)
        svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=2016, shrinking=True,
        tol=0.001, verbose=False)
        subX_test_scale = min_max_scaler.transform(subX_test)
        predict_subY = classifier.predict(subX_test_scale)
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