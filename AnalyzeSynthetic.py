import sys
import time
import pdb
import getpass
import glob
import multiprocessing

from IPython import embed
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sklearn.datasets

import supervised_learning
import parameter_search
import learning.trn_simple_regression 
import learning.trn_aggregation

plt.ion()


LOAD_TEST_DATA = True
CFG = 3
ANALYSIS_VISUALIZATION = False

SCORE_THRESHOLD = 0.40

LEARNER = 'nn_agg'
#LEARNER = 'svr_mean'
USE_CACHE = True

# define a base random seed
np.random.seed(111)


if LOAD_TEST_DATA:
    if CFG == 1:
        N_FE = 3
        N_SA = 1024
        N_TR = 3
        NORMAL_NOISE_STD = 0.45
        EXTRA_PERCENTAGE = 0.25
        EXTRA_NOISE_STD = 1.0
    elif CFG == 2:
        N_FE = 3
        N_SA = 1024
        N_TR = 3
        NORMAL_NOISE_STD = 0.35
        EXTRA_PERCENTAGE = 0.40
        EXTRA_NOISE_STD = 1.0
    elif CFG == 3:
        N_FE = 3
        N_SA = 1024
        N_TR = 3
        NORMAL_NOISE_STD = 0.45
        EXTRA_PERCENTAGE = 0.50
        EXTRA_NOISE_STD = 1.0
    
    np.random.seed(0)
    exp_ids = np.arange(0,N_SA).reshape(-1,1)
    exp_features = np.random.rand(N_SA,N_FE)
    exp_targets = exp_features[:,0] + 3**exp_features[:,1] + \
            exp_features[:,2]**7
    exp_targets = np.hstack([exp_targets.reshape(-1,1)]*N_TR)
    exp_targets = (exp_targets-exp_targets.mean(axis=0)) / \
            exp_targets.std(axis=0)
    # add some gaussian noise
    exp_targets += np.random.normal(loc=0.0, 
            scale=NORMAL_NOISE_STD, size=exp_targets.shape)
    # add extra noise to some trials
    for ind_fe in range(N_FE):
        extra_inds = np.random.choice(N_SA, int(EXTRA_PERCENTAGE*N_SA), 
                replace=False)
        exp_targets[extra_inds,ind_fe] *= np.random.normal(loc=0.0, 
                scale=EXTRA_NOISE_STD, size=(len(extra_inds),))
    #pdb.set_trace()


# train and test
"""
if LEARNER == 'lsearch_expsel':
    theta = parameter_search.lsearch_expsel(features=exp_features, 
            targets=exp_targets, ids=exp_ids)
    accuracies = []
    for ind_trntst in range(20):
        accu = supervised_learning.regress_svr(features=exp_features, 
                targets=exp_targets.dot(theta).ravel(), 
                ids=exp_ids, debug=False)
        accuracies.append(accu)
    accu_mean = {}
    for k in accuracies[0].keys():
        accu_mean[k] = np.mean([a[k] for a in accuracies]) 
    print(accu_mean)
    embed()
elif LEARNER == 'gsearch_expsel':
    parameter_search.gsearch_expsel(features=exp_features, 
            targets=exp_targets, ids=exp_ids)
elif LEARNER == 'svr_fast':
    supervised_learning.regress_svr(features=exp_features, 
            targets=exp_targets.max(axis=1), ids=exp_ids, 
            params={'kernel':['linear'], 'C':10.0**np.linspace(0,2,10)}, 
            n_folds=10)
"""
if LEARNER == 'nn_constweight':
    theta = [0.6567, 0.2227, 0.1205]
    exp_targets = exp_targets.dot(np.vstack(theta))
    #exp_features = np.random.rand(72*3).reshape((-1,3))
    #exp_targets = exp_features[:,0].reshape(-1,1)
    
    accu = learning.trn_simple_regression.regress_nn(
            features=exp_features, 
            targets=exp_targets, ids=exp_ids, 
            params=None, 
            n_folds=10,
            debug=True, seed=0)
elif LEARNER == 'svr_mean':
    agg_targets = np.mean(exp_targets, axis=1)
    accu = supervised_learning.regress_svr(features=exp_features, 
            targets=agg_targets, ids=exp_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1, debug=True)

    #embed()
elif LEARNER == 'nn_agg':
    res = learning.trn_aggregation.regress_nn(
            features=exp_features, 
            targets=exp_targets, ids=exp_ids, 
            params=None, 
            n_folds=5,
            debug=False, seed=-1)
    print(res) 
    agg_preds = res['aggregate_preds']
    agg_targets = np.sum(agg_preds*exp_targets, axis=1)
    accu = supervised_learning.regress_svr(features=exp_features, 
            targets=agg_targets, ids=exp_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1,debug=True)

    #embed()
elif LEARNER == 'nn_agg_active':
    try:
        agg_dataset = scipy.io.loadmat(
                './run_data/agg_dataset.mat')
    except:
        agg_dataset = None
    if (not USE_CACHE) or (type(agg_dataset)==type(None)):
        res = learning.trn_aggregation.regress_nn(
                features=exp_features, 
                targets=exp_targets, ids=exp_ids, 
                params=None, 
                n_folds=5,
                debug=False, seed=-1)
        print(res) 
        agg_preds = res['aggregate_preds']
        scipy.io.savemat('./run_data/agg_dataset.mat', 
                {'agg_preds':agg_preds, 
                'exp_ids':exp_ids,
                'exp_features':exp_features,
                'exp_targets':exp_targets})
    else:
        print('Using cached agg_preds.')
    agg_dataset = scipy.io.loadmat(
            './run_data/agg_dataset.mat')
    exp_ids = agg_dataset['exp_ids']
    exp_targets = agg_dataset['exp_targets']
    exp_features = agg_dataset['exp_features']
    agg_preds = agg_dataset['agg_preds']
    agg_targets = np.sum(agg_preds*exp_targets, axis=1)
    
    accuracies = []
    #for _ in range(10):
    def trn_eval(criteria, seed):
        accu = supervised_learning.regress_active_svr(
                features=exp_features, 
                targets=agg_targets.ravel(), ids=exp_ids, 
                initial_portion=0.20, final_portion=1.00, 
                criteria = criteria, seed=seed, step_size=2,
                debug=False)
        #accuracies.append(accu)
        return accu
    pool = multiprocessing.Pool(8)
    runs = 32 
    accuracies = [pool.apply_async(trn_eval, ('committee',seed))\
            for seed in range(runs)]

    accuracies = [a.get() for a in accuracies]
    accu_mean = {}
    for k in accuracies[0].keys():
        accu_k = [a[k] for a in accuracies]
        accu_mean[k] = np.vstack(accu_k).mean(axis=0)
    
    accuracies = [pool.apply_async(trn_eval, ('rand',seed))\
            for seed in range(runs)]
    accuracies = [a.get() for a in accuracies]
    accu_mean_rand = {}
    for k in accuracies[0].keys():
        accu_k = [a[k] for a in accuracies]
        accu_mean_rand[k] = np.vstack(accu_k).mean(axis=0)
    
    plt.ion()
    plt.figure()
    plt.plot(accu_mean['portions'],accu_mean['portions_r_value'], 'k')
    plt.plot(accu_mean['portions'],accu_mean['portions_r_value'], 'ok')
    plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_r_value'],'k--')
    plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_r_value'],'^k')
    plt.xlabel('portions')
    plt.ylabel('r_value')
    plt.figure()
    plt.plot(accu_mean['portions'],accu_mean['portions_mae'], 'k')
    plt.plot(accu_mean['portions'],accu_mean['portions_mae'], 'ok')
    plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_mae'],'k--')
    plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_mae'],'^k')
    plt.xlabel('portions')
    plt.ylabel('MAE')
    plt.figure()
    plt.plot(accu_mean['portions'],accu_mean['portions_std'], 'k')
    plt.plot(accu_mean['portions'],accu_mean['portions_std'], 'ok')
    plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_std'],'k--')
    plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_std'],'^k')
    plt.xlabel('portions')
    plt.ylabel('STD')
    plt.ion()
    plt.draw()
    embed()
    
else:
    raise ValueError('Invalid LEARNER')

#plt.tight_layout()
#plt.draw()
#embed()
