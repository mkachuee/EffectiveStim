import pdb

from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm

plt.ion()

DEBUG = False

def classify_svm(features, targets, 
        #params={'kernel': ['linear'],'C': 10.0**np.arange(-18,18,4)}, 
        params={'kernel': ['rbf'],'C': 10.0**np.arange(0,6,0.5), 'gamma': 10.0**np.arange(-3,0,0.5)}, 
        n_folds=10):
    # random permutation
    ind_perms = np.random.permutation(features.shape[0])
    features = features[ind_perms]
    targets = targets[ind_perms]
    # normalize features
    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    
    # to k fold test and train
    test_targets = []
    test_predictions = []
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    for ind_train, ind_test in kf.split(targets):
        # create a cross validated model for each fold
        clf = sklearn.model_selection.GridSearchCV(sklearn.svm.SVC(), params, 
                verbose=1, n_jobs=4, cv=n_folds)
        clf.fit(features[ind_train], targets[ind_train])
        test_predictions.append(clf.predict(features[ind_test]))
        test_targets.append(targets[ind_test])
        
    # evaluate the model
    test_targets = np.hstack(test_targets)
    test_predictions = np.hstack(test_predictions)
    accuracy = np.around((test_targets == test_predictions).mean(), decimals=4)
    accuracy_null = np.around(
            max((test_targets==1).mean(),(test_targets==0).mean()), 
            decimals=4)
    print('Accuracy is: ' + str(accuracy*100))
    print('Null accuracy is: ' + str(accuracy_null*100))
    if DEBUG:
        #plt.figure()
        #plt.plot(features)
        embed()
