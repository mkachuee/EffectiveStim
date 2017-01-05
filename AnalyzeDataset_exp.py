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


LOAD_TEST_DATA = False
DATASET_NAME = sys.argv[1]#'dataset_718885'

ANALYSIS_VISUALIZATION = False

ANALYSIS_CLASSIFICATION = True
COMBINE_EXPS = False
TARGET = 'force' 
DIFFERENCE_MAX = 1.5#2.00
DIFFERENCE_MIN = -0.50
VARIATION_MAX = 9991.10
ANOMALITY_MAX = 9991.0

SCORE_THRESHOLD = 0.40

LEARNER = 'nn_agg_active'
#LEARNER = 'svr_comp'
#'nn_agg'#'nn_expsel'#'lsearch_expsel' #'gsearch_expsel'
USE_CACHE = True
BLACKLIST = ['HG17','HG25']

# define a base random seed
np.random.seed(111)

# load dataset file
dataset = scipy.io.loadmat('./run_data/' + DATASET_NAME + '.mat')
dataset_features = dataset['dataset_features']
dataset_targets = dataset['dataset_targets']
dataset_sessions = dataset['dataset_sessions']

# merge sessions
exp_features = [] # it would be list of [Intensity, Location, Frequency]
exp_targets = []
exp_ids = []

if COMBINE_EXPS:
    # for each session
    for (session_features, session_targets, session_names) in \
        zip(dataset_features[0], dataset_targets[0], dataset_sessions[0]):
        # for each block
        baseline_scores = None
        for (block_features, block_targets, block_name) in \
            zip(session_features, session_targets, session_names):
        # check block type
        # if it is baseline
            if block_features[1] == 1:
                base_inds = (session_features[:,2:]==block_features[2:]).all(axis=1)
                baseline_scores = np.median(session_targets[base_inds],axis=0)
                #baseline_scores = block_targets
            elif block_features[1] ==2:
                try:
                    #embed()
                    if not np.all((block_features[2:]==exp_features),axis=-1).any():
                        conf_inds = (session_features[:,2:] == \
                                block_features[2:]).all(axis=1)
                        block_targets = np.median(session_targets[conf_inds],axis=0)
                        #exp_targets.append(block_targets / baseline_scores)
                        exp_targets.append((block_targets-baseline_scores) \
                                / baseline_scores)
                        exp_features.append(block_features[2:])
                        exp_ids.append(block_name)
                except:
                    continue
            else:
                pass
else:
# for each session
    np.random.seed(None)
    ind_perms = np.random.permutation(dataset_features[0].shape[0])
    dataset_features[0] = dataset_features[0][ind_perms]
    dataset_targets[0] = dataset_targets[0][ind_perms]
    dataset_sessions[0] = dataset_sessions[0][ind_perms]

    for (session_features, session_targets, session_names) in \
        zip(dataset_features[0], dataset_targets[0], dataset_sessions[0]):
        # for each block
        baseline_scores = None
        for (block_features, block_targets, block_name) in \
            zip(session_features, session_targets, session_names):
        # check block type
        # if it is baseline
            if block_features[1] == 1:
                base_inds = (session_features[:,2:]==block_features[2:]).all(axis=1)
                baseline_scores = np.median(session_targets[base_inds],axis=0)
                #baseline_scores = block_targets
            elif block_features[1] ==2:
                try:
                        exp_targets.append((block_targets-baseline_scores) \
                                / baseline_scores)
                        exp_features.append(block_features[2:])
                        exp_ids.append(block_name)
                except:
                    continue
            else:
                pass

exp_features = np.vstack(exp_features)
exp_targets = np.vstack(exp_targets)
exp_ids = np.vstack(exp_ids)

# apply variation max
if TARGET == 'force':
    exp_targets = exp_targets[:,:3]
elif TARGET == 'area':    
    exp_targets = exp_targets[:,3:6]
else:
    assert False

tar_var = (np.abs(exp_targets - exp_targets.mean(axis=1).reshape(-1,1)) / \
        exp_targets.mean(axis=1).reshape(-1,1))
inds = tar_var.max(axis=1) < VARIATION_MAX
exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]

# manual feature transforms
exp_features[:,2] = np.log(exp_features[:,2])
# NOTE: here, median works better!
# exp_targets = exp_targets.max(axis=1).reshape(-1,1)
# exp_targets = np.median(exp_targets, axis=1).reshape(-1,1)
exp_targets = exp_targets


# apply difference max and min
inds = ((exp_targets<DIFFERENCE_MAX) * (exp_targets>DIFFERENCE_MIN)).all(
        axis=1).ravel()
exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]


# anomality removal
fe = np.hstack([exp_features,exp_targets])
fe_mean = np.mean(fe, axis=0)
fe_std = np.std(fe, axis=0)
fe_dev = np.sum(((fe - fe_mean)/fe_std)**2, axis=1)/fe.shape[1]

inds = (fe_dev < ANOMALITY_MAX)

exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]


if LOAD_TEST_DATA:
    N_FE = 3
    N_SA = 1024
    N_TR = 3
    NORMAL_NOISE_STD = 0.45
    EXTRA_PERCENTAGE = 0.25
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

for k in BLACKLIST:
    inds_ok = (np.core.defchararray.find(exp_ids,k) < 0).ravel()
    exp_ids = exp_ids[inds_ok]
    exp_targets = exp_targets[inds_ok]
    exp_features = exp_features[inds_ok]


if ANALYSIS_CLASSIFICATION:
    
    # train and test
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
    elif LEARNER == 'nn_expsel':
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
    elif LEARNER == 'svr_comp':
        agg_targets = np.mean(exp_targets, axis=1)
        accu = supervised_learning.regress_svr(features=exp_features, 
                targets=agg_targets, ids=exp_ids, 
                params=[{ 'kernel': ['rbf'],
                    'C': 10.0**np.linspace(0,2,10),
                    'gamma': 10.0**np.linspace(-3,-1,10),
                    'epsilon': 10.0**np.linspace(-3,-1,10),}],
                n_folds=10, debug=True)

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
                    'C': 10.0**np.linspace(0,2,10),
                    'gamma': 10.0**np.linspace(-3,-1,10),
                    'epsilon': 10.0**np.linspace(-3,-1,10),}],
                n_folds=10, debug=True)

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
                    criteria = criteria, seed=seed, step_size=1,
                    debug=False)
            #accuracies.append(accu)
            return accu
        pool = multiprocessing.Pool(8)
        runs = 16 
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
        #plt.plot(accu_mean['portions'],accu_mean['portions_r_value'], 'ok')
        plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_r_value'],'k--')
        #plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_r_value'],'^k')
        plt.xlabel('Fraction of training data')
        plt.ylabel('r_value')
        plt.legend(['Active Learning', 'Random Sampling'], loc='lower right')
        plt.figure()
        plt.plot(accu_mean['portions'],accu_mean['portions_mae'], 'k')
        #plt.plot(accu_mean['portions'],accu_mean['portions_mae'], 'ok')
        plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_mae'],'k--')
        #plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_mae'],'^k')
        plt.xlabel('Fraction of training data')
        plt.ylabel('MAE')
        plt.legend(['Active Learning', 'Random Sampling'], loc='upper right')
        plt.figure()
        plt.plot(accu_mean['portions'],accu_mean['portions_std'], 'k')
        #plt.plot(accu_mean['portions'],accu_mean['portions_std'], 'ok')
        plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_std'],'k--')
        #plt.plot(accu_mean_rand['portions'],accu_mean_rand['portions_std'],'^k')
        plt.xlabel('Fraction of training data')
        plt.ylabel('STD')
        plt.legend(['Active Learning', 'Random Sampling'], loc='upper right')
        plt.ion()
        plt.draw()
        embed()
        
    else:
        raise ValueError('Invalid LEARNER')

#plt.tight_layout()
#plt.draw()
#embed()
