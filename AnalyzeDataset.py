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

import supervised_learning

plt.ion()

DATASET_NAME = sys.argv[1]#'dataset_718885'

ANALYSIS_VISUALIZATION = False

ANALYSIS_CLASSIFICATION = True
TARGET = None # FIXME
SCORE_THRESHOLD = 1.4

LEARNER = 'svm'


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
            baseline_scores = block_targets
        elif block_features[1] ==2:
            try:
                exp_targets.append(block_targets / baseline_scores)
                exp_features.append(block_features[2:])
                #exp_ids.append( TODO
            except:
                continue
        else:
            pass

exp_features = np.vstack(exp_features)
exp_targets = np.vstack(exp_targets)

# manual feature transforms
exp_features[:,2] = np.log(exp_features[:,2])
exp_targets = exp_targets[:,:3].max(axis=1).reshape(-1,1)

# anomality removal
fe = np.hstack([exp_features,exp_targets])
fe_mean = np.mean(fe, axis=0)
fe_std = np.std(fe, axis=0)
fe_dev = np.sum(((fe - fe_mean)/fe_std)**2, axis=1)/fe.shape[1]
inds = (fe_dev < 1.0)

#exp_ids = exp_ids[inds]
exp_targets = exp_targets[inds]
exp_features = exp_features[inds]

# apply threshold
#exp_targets_classes = (exp_targets[:,TARGET] > SCORE_THRESHOLD).astype(np.int)
exp_targets_classes = (exp_targets > SCORE_THRESHOLD).astype(np.int)
exp_targets_classes = exp_targets_classes.ravel()
 #exp_targets_classes = (exp_targets[:,3:].max(axis=1) > SCORE_THRESHOLD).astype(np.int)

# some outlier checks
# FIXME: uncomment after investigation
#valid_exps = ((exp_targets < 1.6) * (exp_targets > 0.8)).all(axis=1)
#exp_features = exp_features[valid_exps]
#exp_targets = exp_targets[valid_exps]

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




if ANALYSIS_CLASSIFICATION:
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

    
    # train and test classifier
    if LEARNER == 'knn':
        supervised_learning.classify_knn(features=exp_features, 
                targets=exp_targets_classes)
    elif LEARNER == 'svm':
        supervised_learning.classify_svm(features=exp_features, 
                targets=exp_targets_classes)

plt.tight_layout()
plt.draw()
embed()
