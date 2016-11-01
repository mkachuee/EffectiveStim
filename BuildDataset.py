import sys
import time
import pdb
import getpass
import glob

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import neo.io.tdtio as tdtio
import process_block


plt.ion()

SUBJECT_NAME = '718885'
#DB_PATH = '/home/' + getpass.getuser() + '/Database/' + SUBJECT_NAME +'/'
DB_PATH = '/media/' + getpass.getuser() + '/Data/Database/' + SUBJECT_NAME +'/'


note_files = glob.glob(DB_PATH + 'Note/*.csv')
data_files = glob.glob(DB_PATH + 'Data/*/')

dataset_sessions = []
dataset_features = []
dataset_targets = []
for note_file in note_files:
    note_file.replace('-', '_')
    session_name = note_file.split('/')[-1].split('.')[-2]
    print('Processing: ' + session_name)
    
    if session_name in ['718885_2016-09-15_week73_HG19']:
        print('Processing :' + session_name + '... SKIPPED')
        continue
    
    #if session_name in ['718885_2016-07-28_week66_HG5','718885_2016-09-22_week74_HG21',
    #        '718885_2016-08-10_week68_HG8', '718885_2016-07-15_week64_HG3', 
    #        '718885_2016-09-01_week71_HG15', '718885_2016-09-29_week75_HG23', 
    #        '718885_2016-08-17_week69_HG10', '718885_2016-07-27_week66_HG4',
    #        '718885_2016-09-08_week72_HG17', '718885_2016-08-04_week67_HG7', 
    #        '718885_2016-07-12_week64_HG1', '718885_2016-09-07_week72_HG16', 
    #        '718885_2016-09-15_week73_HG19', '718885_2016-08-11_week68_HG9', 
    #        '718885_2016-08-18_week69_HG11', '718885_2016-07-14_week64_HG2', 
    #        '718885_2016-08-03_week67_HG6']:
    #    print('Processing :' + session_name + '... SKIPPED')
    #    continue

    try:
        session_features = np.loadtxt(note_file, delimiter=',', 
                comments='#')
    except:
        print('WARNING: csv load failed.')
        continue
    
    # for each block load the signals
    session = tdtio.TdtIO(dirname=DB_PATH+'Data/' + session_name + '/' + \
            session_name)
    session_features_matched = []
    session_targets_matched = []
    for session_feature in session_features:
        try:
            # load raw block data
            block_name = 'Block-' + str(int(session_feature[0]))
            block_data = session.read_block_improved(blockname=block_name)
        except:
            print('WARNING: tdt/sev load failed.')
            continue

        # extract target values from the block
        print('Processing: ' + session_name + ' ' +block_name)
        session_target = process_block.extract_targets(block_data)#, debug=True)
        #try:
        #    #session_target = process_block.extract_targets(block_data, debug=True)
        if session_target is None:
            print('WARNING: extract target failed.')
            continue
        session_features_matched.append(session_feature)
        session_targets_matched.append(session_target)
        
        #except:
        #    print('WARNING: extract target failed.')
        #    continue
    
    try:
        # append features and targets, etc.
        dataset_features.append(np.vstack(session_features_matched))
        dataset_targets.append(np.vstack(session_targets_matched))
        dataset_sessions.append(session_name)    
        print('Processing :' + session_name + '... DONE!')
        print(79*'-')
    except:    
        print('WARNING: session failed.')
        print(79*'-')
        continue
    

    scipy.io.savemat('./run_data/dataset_'+SUBJECT_NAME+'.mat', 
            {'dataset_sessions':dataset_sessions, 
                'dataset_features':dataset_features, 
                'dataset_targets':dataset_targets})


embed()
