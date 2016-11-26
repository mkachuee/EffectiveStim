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

plt.ion()

DATASET_NAME = sys.argv[1]#'dataset_718885'

ANALYSIS_VISUALIZATION = False

ANALYSIS_CLASSIFICATION = True
TARGET = 'force' 
DIFFERENCE_MAX = 1.5#2.00
DIFFERENCE_MIN = -0.50
VARIATION_MAX = 9990.10
ANOMALITY_MAX = 9991.0
LOAD_TEST_DATA = False

SCORE_THRESHOLD = 0.40

LEARNER = 'svr'# knn, svc, svr


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
for (session_features, session_targets) in \
    zip(dataset_features[0], dataset_targets[0]):
    # for each block
    baseline_scores = None
    for (block_features, block_targets) in \
        zip(session_features, session_targets):
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
                    #pdb.set_trace()
                    block_targets = np.median(session_targets[conf_inds],axis=0)
                    #exp_targets.append(block_targets / baseline_scores)
                    exp_targets.append((block_targets-baseline_scores) \
                            / baseline_scores)
                    exp_features.append(block_features[2:])
                    #exp_ids.append( TODO
            except:
                continue
        else:
            pass

exp_features = np.vstack(exp_features)
exp_targets = np.vstack(exp_targets)



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
#exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]

# manual feature transforms
exp_features[:,2] = np.log(exp_features[:,2])
exp_targets = exp_targets.max(axis=1).reshape(-1,1)

# apply difference max and min
inds = ((exp_targets<DIFFERENCE_MAX) * (exp_targets>DIFFERENCE_MIN)).ravel()
#exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]


# anomality removal
fe = np.hstack([exp_features,exp_targets])
fe_mean = np.mean(fe, axis=0)
fe_std = np.std(fe, axis=0)
fe_dev = np.sum(((fe - fe_mean)/fe_std)**2, axis=1)/fe.shape[1]

inds = (fe_dev < ANOMALITY_MAX)
#inds = (fe_dev < 1.0) * (exp_targets.ravel() < 2.0) * \
#        (exp_targets.ravel() > 0.8)

#exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]


# apply threshold
exp_targets_classes = (exp_targets > SCORE_THRESHOLD).astype(np.int)
exp_targets_classes = exp_targets_classes.ravel()


if ANALYSIS_VISUALIZATION:
     
    # visualize the experimental data
    x = exp_features[:,1]
    y = exp_features[:,2]
    z = exp_targets[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=np.abs(z), marker='o')

    ax.set_xlabel('Location')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Score')
    plt.title('Score(L,F) at different Intensities' )

    # visualize the experimental data
    x = exp_features[:,1]
    y = exp_features[:,0]
    z = exp_targets[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=np.abs(z), marker='o')

    ax.set_xlabel('Location')
    ax.set_ylabel('Intensity')
    ax.set_zlabel('Score')
    plt.title('Score(L,F) at different frequencies')

    # visualize the experimental data
    x = exp_features[:,0]
    y = exp_features[:,1]
    z = exp_features[:,2]
    c = exp_targets[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=np.abs(c), marker='o', cmap=cm.jet)
    
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Location')
    ax.set_zlabel('Frequency')
    plt.title('Score is indicated by color')

    # visualize score classes on exp data
    x = exp_features[:,0]
    y = exp_features[:,1]
    z = exp_features[:,2]
    t = exp_targets_classes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=t, marker='o', cmap=cm.prism)#, cmap=cm.jet)
        
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Location')
    ax.set_zlabel('Frequency')
    plt.title('Score class is indicated by color')

if LOAD_TEST_DATA:
    exp_features, exp_targets = sklearn.datasets.load_diabetes(True)
        

if ANALYSIS_CLASSIFICATION:
    
    # train and test classifier
    if LEARNER == 'knn':
        supervised_learning.classify_knn(features=exp_features, 
                targets=exp_targets_classes)
    elif LEARNER == 'svc':
        supervised_learning.classify_svm(features=exp_features, 
                targets=exp_targets_classes)
    elif LEARNER == 'svr':
        supervised_learning.regress_svr(features=exp_features, 
                targets=exp_targets.ravel())


#plt.tight_layout()
plt.draw()
#embed()
