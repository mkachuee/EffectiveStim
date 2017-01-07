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

import learning.supervised_learning
import learning.nn_trainer

plt.ion()


TEST_DATASET = 'boston'#'bp'#'diabetes'
CFG = 1
ANALYSIS_VISUALIZATION = False



params_nn = {
        'N_FE':3,
        'N_SN':3,
        'SIZE_BATCH':0.9,
        'SIZE_HIDDENS':[32],
        'SIZE_HIDDENS_AGG':[32],
        'RATE_LEARNING_1':1.0e-1,
        'RATE_LEARNING_AGG_1':1.0e-1,
        'RATE_LEARNING_2':1.0e-2,
        'RATE_LEARNING_AGG_2':1.0e-2,
        'MAX_STEPS':100000,
        'MAX_EARLYSTOP':10,
        'DIR_LOG':'./logs'}

# define a base random seed
np.random.seed(111)

def synthesize_noise(targets, params, seed=0):
    np.random.seed(seed)
    exp_targets = np.hstack([targets.reshape(-1,1)]*params['N_SN'])
    exp_targets = (exp_targets-exp_targets.mean(axis=0)) / \
            exp_targets.std(axis=0)
    # add some gaussian noise
    exp_targets += np.random.normal(loc=0.0, 
            scale=params['NORMAL_NOISE_STD'], size=exp_targets.shape)
    # add extra noise to some trials
    for ind_sn in range(params['N_SN']):
        extra_inds = np.random.choice(exp_targets.shape[0], 
                int(params['EXTRA_PERCENTAGE']*exp_targets.shape[0]), 
                replace=False)
        exp_targets[extra_inds,ind_sn] += np.random.normal(loc=0.0, 
                scale=params['EXTRA_NOISE_STD'], size=(len(extra_inds),))
    #pdb.set_trace()
    return exp_targets

# load dataset
if TEST_DATASET == 'diabetes':
    exp_dataset = sklearn.datasets.load_diabetes()
    exp_features = exp_dataset.data
    exp_targets = exp_dataset.target.reshape(-1,1)
    exp_ids = np.arange(0,exp_targets.shape[0]).reshape(-1,1)
elif TEST_DATASET == 'boston':
    exp_dataset = sklearn.datasets.load_boston()
    exp_features = exp_dataset.data
    exp_targets = exp_dataset.target.reshape(-1,1)
    exp_ids = np.arange(0,exp_targets.shape[0]).reshape(-1,1)
elif TEST_DATASET == 'bp':
    exp_dataset = scipy.io.loadmat('./run_data/dataset_BP.mat')
    exp_features = exp_dataset['dataset'][0]['features'][0]
    exp_targets = exp_dataset['dataset'][0]['systolic'][0]
    exp_ids = exp_dataset['dataset'][0]['id'][0].reshape(-1,1)

# simulate using synthesized noise
syn_targets = synthesize_noise(targets=exp_targets, 
        params={
            'N_SN':3,
            'NORMAL_NOISE_STD':0.05,
            'EXTRA_PERCENTAGE':0.50,
            'EXTRA_NOISE_STD':1.0})

dataset_config = [{'ids':exp_ids, 
        'features':exp_features, 
        'targets':syn_targets}]

for cfg in dataset_config:
    cfg_ids = cfg['ids']
    cfg_features = cfg['features']
    cfg_targets = cfg['targets']
    
    # train and test each method
    # mean
    agg_targets = np.mean(cfg_targets, axis=1)
    accu_mean = learning.supervised_learning.regress_svr(
            features=cfg_features, 
            targets=agg_targets, ids=cfg_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1, debug=False)
    
    # median
    agg_targets = np.median(cfg_targets, axis=1)
    accu_median = learning.supervised_learning.regress_svr(
            features=cfg_features, 
            targets=agg_targets, ids=cfg_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1, debug=False)
    
    # dynamic weight prediction
    params_nn['N_FE'] = cfg_features.shape[1]
    params_nn['N_SN'] = cfg_targets.shape[1]
    trainer = learning.nn_trainer.NNTrainer(params=params_nn)
    res = trainer.regress_nn(
            features=cfg_features, 
            targets=cfg_targets, ids=cfg_ids, 
            params=None, 
            n_folds=5,
            debug=False, seed=-1)
    print(res) 
    agg_preds = res['aggregate_preds']
    agg_targets = np.sum(agg_preds*cfg_targets, axis=1)
    accu_agg = learning.supervised_learning.regress_svr(
            features=cfg_features, 
            targets=agg_targets, ids=cfg_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1,debug=False)

    

#plt.tight_layout()
#plt.draw()
embed()
