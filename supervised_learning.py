import pdb

from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import scipy

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import sklearn.neighbors

plt.ion()

DEBUG = True

def classify_svm(features, targets, 
        #params={'kernel': ['linear'],'C': 10.0**np.arange(-18,18,4)}, 
        params={'kernel': ['rbf'],'C': 10.0**np.arange(0,6,0.5), 'gamma': 10.0**np.arange(-5,1,0.5)}, 
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
                verbose=1, n_jobs=8, cv=n_folds)
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


def classify_knn(features, targets, 
        #params={'kernel': ['linear'],'C': 10.0**np.arange(-18,18,4)}, 
        params={'n_neighbors':np.arange(1,20)}, 
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
        clf = sklearn.model_selection.GridSearchCV(sklearn.neighbors.KNeighborsClassifier(), 
                params, verbose=1, n_jobs=8, cv=n_folds)
        #clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=params['k'], 
        #        n_jobs=8)
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


def regress_svr(features, targets, ids, 
        #params={'kernel': ['linear'],'C': 10.0**np.arange(-18,18,4)}, 
        params= [
            {
            'kernel': ['rbf'],
            'C': 10.0**np.linspace(0,2,10), 
            'gamma': 10.0**np.linspace(-3,-1,10),
            'epsilon': 10.0**np.linspace(-4,-2,10),
            #'kernel': ['rbf'],
            #'C': 10.0**np.arange(-1,3,0.5), 
            #'gamma': 10.0**np.arange(-2,1,0.5),
            #'epsilon': 10.0**np.arange(-2,1,0.5)
            }, 
            #{
            #'kernel': ['poly'],
            #'C': 10.0**np.arange(-1,3,1), 
            #'gamma': 10.0**np.arange(-2,1,1),
            #'epsilon': 10.0**np.arange(-2,1,1),
            #'degree': np.arange(2,5,1)
            #}
            ], 
        n_folds=10, 
        debug=False):
    # random permutation
    ind_perms = np.random.permutation(features.shape[0])
    features = features[ind_perms]
    targets = targets[ind_perms]
    ids = ids[ind_perms]
    # normalize features
    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    
    # to k fold test and traini
    test_ids = []
    test_targets = []
    test_predictions = []
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    for ind_train, ind_test in kf.split(targets):
        # create a cross validated model for each fold
        clf = sklearn.model_selection.GridSearchCV(sklearn.svm.SVR(), params, 
                verbose=0, n_jobs=8, cv=n_folds)
        clf.fit(features[ind_train], targets[ind_train])
        test_predictions.append(clf.predict(features[ind_test]))
        test_targets.append(targets[ind_test])
        test_ids.append(ids[ind_test])
    # evaluate the model
    test_targets = np.hstack(test_targets)
    test_predictions = np.hstack(test_predictions)
    test_ids = np.vstack(test_ids)
    mae = np.around((np.abs(test_targets-test_predictions)).mean(), 
            decimals=4)
    std = np.around(np.std(test_targets-test_predictions), 
            decimals=4)
    mae_null = np.around((np.abs(test_targets-test_targets.mean())).mean(), 
            decimals=4)
    std_null = np.around(np.std(test_targets-test_targets.mean()), 
            decimals=4)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            test_targets,test_predictions) 
    if debug:
        print('MAE is: ' + str(mae))
        print('Null MAE is: ' + str(mae_null))
        print('STD is: ' + str(std))
        print('Null STD is: ' + str(std_null))
        print('r is: ' + str(r_value))
        #print('Null r is: ' + str(r_value_null))
        plt.figure()
        plt.plot(test_targets,test_predictions, 'o')
        reg_range = np.linspace(np.min(test_targets), 
                np.max(test_targets), 100)
        plt.plot(reg_range,reg_range, '.')
        plt.plot(reg_range, r_value*reg_range+intercept)
        plt.xlabel('Target')
        plt.ylabel('Prediction')
        plt.title('r = '+str(np.around(r_value, 2)) + \
                '\n MAE = '+str(100*mae) + ', STD= '+str(100*std))
        plt.axis('equal')
        embed()
    
    return {'r_value':r_value, 'MAE':mae, 'STD':std}


def regress_exp(features, targets, ids, 
        #params={'kernel': ['linear'],'C': 10.0**np.arange(-18,18,4)}, 
        params= [
            {
            'kernel': ['rbf'],
            'C': 10.0**np.linspace(0,2,10), 
            'gamma': 10.0**np.linspace(-3,-1,10),
            'epsilon': 10.0**np.linspace(-4,-2,10),
            #'kernel': ['rbf'],
            #'C': 10.0**np.arange(-1,3,0.5), 
            #'gamma': 10.0**np.arange(-2,1,0.5),
            #'epsilon': 10.0**np.arange(-2,1,0.5)
            }, 
            #{
            #'kernel': ['poly'],
            #'C': 10.0**np.arange(-1,3,1), 
            #'gamma': 10.0**np.arange(-2,1,1),
            #'epsilon': 10.0**np.arange(-2,1,1),
            #'degree': np.arange(2,5,1)
            #}
            ], 
        n_folds=10):
    

    # calc initial prediction value
    pred = np.median(targets, axis=1)
    # targets * theta = pred, theta?
    theta = np.linalg.lstsq(targets, pred)[0]
    pred_new = (targets.dot(theta.reshape(-1,1))) 
    # create a model to predict theta using norm_scores(s)

    # use the model to train a pred estimator
    # use new pred to calc new theta
    # repeat!



    embed()

