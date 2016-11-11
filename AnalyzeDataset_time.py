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


DATASET_NAME = sys.argv[1]

#DATASET_NAME = 'dataset_718885'

ANALYSIS_VISUALIZATION = True

ANALYSIS_CLASSIFICATION = True
TARGET = 0
SCORE_THRESHOLD = 1.2

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
    for (block_features, block_targets, block_name) in \
        zip(session_features, session_targets, session_names):
        if True:#try:
            exp_targets.append(block_targets)
            exp_features.append(block_features[2:])
            exp_ids.append(block_name)
        #except:
        #    continue

exp_features = np.vstack(exp_features)
exp_targets = np.vstack(exp_targets)
exp_ids = np.vstack(exp_ids)



plt.plot(exp_targets[:,:3].max(axis=1))
plt.xticks(10*np.arange(len(exp_ids[::10])), exp_ids[::10], 
        rotation=90, fontsize=6)


plt.tight_layout()
plt.pause(0.01)
plt.draw()
embed()
