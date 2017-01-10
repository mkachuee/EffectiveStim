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


TEST_DATASET = 'diabetes'#'bp'#'diabetes'
STEPS = 10
STEP_BASE = 0
#STEP_BASE = 1#0
STEP_SIZE = 0.10
#STEP_SIZE = 1#0.07#1
#SWEEP_PARAM = 'n_sn'#'prob_extra'#'std_extra'
SWEEP_PARAM = 'prob_extra'

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
        'MAX_EARLYSTOP':2,
        'DIR_LOG':'./logs'}

# define a base random seed
np.random.seed(111)

def synthesize_noise(targets, params, seed=0):
    np.random.seed(seed)
    exp_targets = np.hstack([targets.reshape(-1,1)]*params['N_SN'])
    exp_targets = (exp_targets-exp_targets.mean(axis=0)) / \
            exp_targets.std(axis=0)
    exp_targets_true = exp_targets.mean(axis=1)
    # add some gaussian noise
    exp_targets += np.random.normal(loc=0.0, 
            scale=params['NORMAL_NOISE_STD'], size=exp_targets.shape)
    # add extra noise to some trials
    for ind_sn in range(params['N_SN']):
        extra_inds = np.random.choice(exp_targets.shape[0], 
                int(params['EXTRA_PERCENTAGE']*exp_targets.shape[0]), 
                replace=False)
        exp_targets[extra_inds,ind_sn] += np.random.normal(loc=0.0, 
                scale=params['EXTRA_NOISE_STD']+1.0e-12, 
                size=(len(extra_inds),))
    #pdb.set_trace()
    return exp_targets, exp_targets_true



base_config = {
    'dataset':'boston',
    'n_sn':3,
    'std_normal':0.05,
    'prob_extra':0.50,
    'std_extra':5.0}
dataset_configs = []
sweep_values = []
for ind_cfg in range(STEPS):
    dataset = TEST_DATASET
    param_sweep = SWEEP_PARAM
    sweep_values.append(STEP_SIZE*ind_cfg+STEP_BASE)
    base_config[param_sweep] = sweep_values[-1]
    base_config['dataset'] = dataset
    dataset_configs.append(base_config.copy())

list_results = []
for cfg in dataset_configs:
    # load dataset
    if cfg['dataset'] == 'diabetes':
        exp_dataset = sklearn.datasets.load_diabetes()
        exp_features = exp_dataset.data
        exp_targets = exp_dataset.target.reshape(-1,1)
        exp_ids = np.arange(0,exp_targets.shape[0]).reshape(-1,1)
    elif cfg['dataset'] == 'boston':
        exp_dataset = sklearn.datasets.load_boston()
        exp_features = exp_dataset.data
        exp_targets = exp_dataset.target.reshape(-1,1)
        exp_ids = np.arange(0,exp_targets.shape[0]).reshape(-1,1)
    elif cfg['dataset'] == 'bp':
        exp_dataset = scipy.io.loadmat('./run_data/dataset_BP.mat')
        exp_features = exp_dataset['dataset'][0]['features'][0]
        exp_targets = exp_dataset['dataset'][0]['systolic'][0]
        exp_ids = exp_dataset['dataset'][0]['id'][0].reshape(-1,1)
    elif cfg['dataset'] == 'rand':
        exp_features = np.random.rand(224,5)
        exp_targets = np.random.rand(224,1)
        exp_ids = np.arange(0,exp_targets.shape[0]).reshape(-1,1)

    # simulate using synthesized noise
    exp_targets, exp_targets_true = synthesize_noise(targets=exp_targets, 
        params={
            'N_SN':cfg['n_sn'],
            'NORMAL_NOISE_STD':cfg['std_normal'],
            'EXTRA_PERCENTAGE':cfg['prob_extra'],
            'EXTRA_NOISE_STD':cfg['std_extra']})

    # train and test each method
    # mean
    agg_targets = np.mean(exp_targets, axis=1)
    accu_mean = {}
    accu_mean['MAE'] = np.abs(agg_targets-exp_targets_true).mean()
    accu_mean['STD'] = np.abs(agg_targets-exp_targets_true).std()
    accu_mean['r_value'] = np.sum(
            (agg_targets-agg_targets.mean())*\
                    (exp_targets_true-exp_targets_true.mean()))
    accu_mean['r_value'] /= np.sqrt(
            np.sum((agg_targets-agg_targets.mean())**2))*\
                    np.sqrt(
                    np.sum((exp_targets_true-exp_targets_true.mean())**2))
    
    """
    accu_mean = learning.supervised_learning.regress_svr(
            features=exp_features, 
            targets=agg_targets, ids=exp_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1, debug=False)
    """
    # median
    agg_targets = np.median(exp_targets, axis=1)
    accu_median = {}
    accu_median['MAE'] = np.abs(agg_targets-exp_targets_true).mean()
    accu_median['STD'] = np.abs(agg_targets-exp_targets_true).std()
    accu_median['r_value'] = np.sum(
            (agg_targets-agg_targets.mean())*\
                    (exp_targets_true-exp_targets_true.mean()))
    accu_median['r_value'] /= np.sqrt(
            np.sum((agg_targets-agg_targets.mean())**2))*\
                    np.sqrt(
                    np.sum((exp_targets_true-exp_targets_true.mean())**2))
    """
    accu_median = learning.supervised_learning.regress_svr(
            features=exp_features, 
            targets=agg_targets, ids=exp_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1, debug=False)
    """
    # dynamic weight prediction
    params_nn['N_FE'] = exp_features.shape[1]
    params_nn['N_SN'] = exp_targets.shape[1]
    trainer = learning.nn_trainer.NNTrainer(params=params_nn)
    
    res = trainer.regress_nn(
            features=exp_features, 
            targets=exp_targets, ids=exp_ids, 
            params=None, 
            n_folds=5,
            debug=False, seed=-1)
    print(res) 
    agg_preds = res['aggregate_preds']
    agg_targets = np.sum(agg_preds*exp_targets, axis=1)
    accu_agg = {}
    accu_agg['MAE'] = np.abs(agg_targets-exp_targets_true).mean()
    accu_agg['STD'] = np.abs(agg_targets-exp_targets_true).std()
    accu_agg['r_value'] = np.sum(
            (agg_targets-agg_targets.mean())*\
                    (exp_targets_true-exp_targets_true.mean()))
    accu_agg['r_value'] /= np.sqrt(
            np.sum((agg_targets-agg_targets.mean())**2))*\
                    np.sqrt(
                    np.sum((exp_targets_true-exp_targets_true.mean())**2))
    
    """
    accu_agg = learning.supervised_learning.regress_svr(
            features=exp_features, 
            targets=agg_targets, ids=exp_ids, 
            params=[{ 'kernel': ['rbf'],
                'C': 10.0**np.linspace(0,2,5),
                'gamma': 10.0**np.linspace(-3,-1,5),
                'epsilon': 10.0**np.linspace(-3,-1,5),}],
            n_folds=5, seed=-1,debug=False)
    """
    list_results.append({'accu_median':accu_median, 
        'accu_mean':accu_mean, 'accu_agg':accu_agg})


maes_mean = [a['accu_mean']['MAE'] for a in list_results]
maes_median = [a['accu_median']['MAE'] for a in list_results]
maes_agg = [a['accu_agg']['MAE'] for a in list_results]

stds_mean = [a['accu_mean']['STD'] for a in list_results]
stds_median = [a['accu_median']['STD'] for a in list_results]
stds_agg = [a['accu_agg']['STD'] for a in list_results]


rs_mean = [a['accu_mean']['r_value'] for a in list_results]
rs_median = [a['accu_median']['r_value'] for a in list_results]
rs_agg = [a['accu_agg']['r_value'] for a in list_results]


font_params = {'fontsize':16}
        #{'fontname':'Times New Roman', 'fontsize':16}#, 
        #'fontweight':'bold'}

plt.figure()
plt.plot(sweep_values, maes_agg, 'k', linewidth=2.0)
plt.plot(sweep_values, maes_median, 'k--', linewidth=2.0)
plt.plot(sweep_values, maes_mean, 'k-.', linewidth=2.0)
plt.legend(['agg', 'median', 'mean'], loc='lower right')
if SWEEP_PARAM == 'prob_extra':
    plt.xlabel('Probability of Contamination', **font_params)
elif SWEEP_PARAM == 'std_extra':
    plt.xlabel('STD of Contamination', **font_params)
elif SWEEP_PARAM == 'n_sn':
    plt.xlabel('Number of Samples', **font_params)
plt.ylabel('MAE', **font_params)
plt.tight_layout()

plt.figure()
plt.plot(sweep_values, stds_agg, 'k', linewidth=2.0)
plt.plot(sweep_values, stds_median, 'k--', linewidth=2.0)
plt.plot(sweep_values, stds_mean, 'k-.', linewidth=2.0)
plt.legend(['agg', 'median', 'mean'], loc='lower right')
if SWEEP_PARAM == 'prob_extra':
    plt.xlabel('Probability of Contamination', **font_params)
elif SWEEP_PARAM == 'std_extra':
    plt.xlabel('STD of Contamination', **font_params)
elif SWEEP_PARAM == 'n_sn':
    plt.xlabel('Number of Samples', **font_params)
plt.ylabel('STD', **font_params)
plt.tight_layout()


plt.figure()
plt.plot(sweep_values, rs_agg, 'k', linewidth=2.0)
plt.plot(sweep_values, rs_median, 'k--', linewidth=2.0)
plt.plot(sweep_values, rs_mean, 'k-.', linewidth=2.0)
plt.legend(['agg', 'median', 'mean'], loc='upper right')
if SWEEP_PARAM == 'prob_extra':
    plt.xlabel('Probability of Contamination', **font_params)
elif SWEEP_PARAM == 'std_extra':
    plt.xlabel('STD of Contamination', **font_params)
elif SWEEP_PARAM == 'n_sn':
    plt.xlabel('Number of Samples', **font_params)
plt.ylabel('r', **font_params)


plt.tight_layout()
plt.draw()
embed()
