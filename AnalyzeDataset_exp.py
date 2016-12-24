import sys
import time
import pdb
import getpass
import glob

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

DATASET_NAME = sys.argv[1]#'dataset_718885'

ANALYSIS_VISUALIZATION = False

ANALYSIS_CLASSIFICATION = True
TARGET = 'force' 
DIFFERENCE_MAX = 1.5#2.00
DIFFERENCE_MIN = -0.50
VARIATION_MAX = 9991.10
ANOMALITY_MAX = 9991.0
LOAD_TEST_DATA = False

SCORE_THRESHOLD = 0.40

LEARNER = 'nn_agg'#'nn_expsel'#'lsearch_expsel' #'gsearch_expsel'


# load dataset file
dataset = scipy.io.loadmat('./run_data/' + DATASET_NAME + '.mat')
dataset_features = dataset['dataset_features']
dataset_targets = dataset['dataset_targets']
dataset_sessions = dataset['dataset_sessions']

# merge sessions
exp_features = [] # it would be list of [Intensity, Location, Frequency]
exp_targets = []
exp_ids = []
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
    exp_features, exp_targets = sklearn.datasets.load_diabetes(True)
        

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
    elif LEARNER == 'nn_agg':
        # FIXME
        #theta = [0.6567, 0.2227, 0.1205]
        #exp_targets = exp_targets.dot(np.vstack(theta))
        #exp_targets = np.matlib.repmat(exp_targets, 1,3)
        accu = learning.trn_aggregation.regress_nn(
                features=exp_features, 
                targets=exp_targets, ids=exp_ids, 
                params=None, 
                n_folds=10,
                debug=True, seed=None)
    else:
        raise ValueError('Invalid LEARNER')

#plt.tight_layout()
#plt.draw()
#embed()
