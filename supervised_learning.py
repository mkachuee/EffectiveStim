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
            'C': 10.0**np.linspace(0,2,3), 
            'gamma': 10.0**np.linspace(-2,-1,3),
            'epsilon': 10.0**np.linspace(-2,-1,3),
            #'kernel': ['rbf'],
            #'C': 10.0**np.linspace(0,2,10), 
            #'gamma': 10.0**np.linspace(-3,-1,10),
            #'epsilon': 10.0**np.linspace(-4,-2,10),
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

def regress_active_svr(features, targets, ids, 
        params= [
            {
            'kernel': ['rbf'],
            'C': 10.0**np.linspace(0,2,3), 
            'gamma': 10.0**np.linspace(-2,-1,3),
            'epsilon': 10.0**np.linspace(-2,-1,3),
            }], 
        n_folds=5, criteria = 'committee', seed= None,  
        initial_portion = 0.25, final_portion = 0.50,
        step_size=2, 
        debug=True):
    # random permutation
    np.random.seed(seed)
    ind_perms = np.random.permutation(features.shape[0])
    features = features[ind_perms]
    targets = targets[ind_perms]
    ids = ids[ind_perms]
    # normalize features
    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    
    # k fold test and training
    test_ids = []
    test_targets = []
    test_predictions = []
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    accuracy_portions = []
    fold_cnt = 1
    for ind_train, ind_test in kf.split(targets):
        print(79*'-')
        print('fold: '+str(fold_cnt))
        fold_cnt += 1
        # create a cross validated model for each fold
        rgr_commitee = [\
            sklearn.model_selection.GridSearchCV(sklearn.svm.SVR(), 
                params, verbose=0, n_jobs=1, cv=n_folds)
            for _ in range(8)]
        
        # train set known indexes
        inds_known = []

        # select a random set of samples to start
        inds_known = ind_train[:int(len(ind_train)*initial_portion)].tolist()
        accu_portions = []  
        current_portion = float(len(inds_known))/len(ind_train)
        while final_portion > current_portion:
            #
            current_portion = float(len(inds_known))/len(ind_train)
            
            # add new samples to known set
            inds_unknown = list(set(ind_train.tolist()).difference(inds_known))
            # update the models
            for rgr in rgr_commitee:
                inds_known_bag = np.random.choice(inds_known, 
                        size=int(len(inds_known)*0.8), replace=True)
                rgr.fit(features[inds_known_bag], targets[inds_known_bag])
            
            if len(inds_unknown) != 0:
                # decide on which samples to ask
                preds_commitee = []
                for rgr in rgr_commitee:
                    preds_commitee.append(rgr.predict(features[inds_unknown]))
            
                # calculate committee variance
                preds_commitee = np.vstack(preds_commitee)
                preds_commitee_norm = preds_commitee/preds_commitee.max(axis=0)
                preds_commitee_std = np.vstack(preds_commitee_norm).std(axis=0)
                inds_request_order = [inds_unknown[r] for r in \
                        np.argsort(preds_commitee_std)[::-1]]
                
                # find best request while preserving random state
                inds_request_rand = np.random.choice(inds_unknown, 
                        size=min(len(inds_unknown),step_size),
                        replace=False).tolist()
                inds_request_comm = inds_request_order[:step_size]
                if criteria == 'committee':
                    inds_request = inds_request_comm
                elif criteria == 'rand':
                    inds_request = inds_request_rand
                else:
                    raise NameError('Unknown AL criteria')
                inds_known += inds_request
            # display stat
            print(40*'-')
            print('Portion: ' + str(current_portion))

            preds_commitee = []
            for rgr in rgr_commitee:
                preds_commitee.append(rgr.predict(features[ind_test]))
            preds_commitee_mean = np.vstack(preds_commitee).mean(axis=0)
            test_predictions = preds_commitee_mean
            test_targets = targets[ind_test]
            test_ids = ids[ind_test]
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
            accu_portions.append({'r_value':r_value, 'MAE':mae, 'STD':std, 
                'portion':current_portion})
            print('Performance: ' + str(accu_portions[-1]))
        # add portion accuracies for this fold
        accuracy_portions.append(accu_portions)
        #if fold_cnt == 2:
        #    print('A'*80)
        #    print(accuracy_portions)
        #    pdb.set_trace()

    portions = [a['portion'] for a in accuracy_portions[0]]
    portions_mae = []
    portions_std = []
    portions_r_value = []
    for portion in portions:
        maes = []
        stds = []
        r_values = []
        for fold in accuracy_portions:
            for step in fold:
                if np.abs(step['portion']-portion) < 0.02:
                #if step['portion'] == portion:
                    maes.append(step['MAE'])
                    stds.append(step['STD'])
                    r_values.append(step['r_value'])
        portions_mae.append(np.mean(maes))
        portions_std.append(np.mean(stds))
        portions_r_value.append(np.mean(r_values))
        
    
    return {'portions_mae':portions_mae, 'portions_std':portions_std, 
            'portions_r_value':portions_r_value, 'portions':portions} 

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

