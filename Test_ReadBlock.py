import sys
import time
import pdb
import getpass
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

import neo.io.tdtio as tdtio

plt.ion()

#DB_PATH = '/home/' + getpass.getuser() + '/Database/718885/'
DB_PATH = '/media/mohammad/Data/Database/718885/'
session_name = '718885_2016-10-26_week79_HG29/'
block_name = 'Block-3'

session = tdtio.TdtIO(dirname=DB_PATH+'Data/' + 2*session_name)
block_data = session.read_block_improved(blockname=block_name)


plt.figure()
subplot_idx = 0
for channel_name in block_data.keys():
    print(channel_name)
    subplot_idx += 1
    plt.subplot(22,1,subplot_idx)
    if channel_name == 'sev':
        len_total = len(block_data[channel_name][0])
        #plt.plot(block_data[channel_name][np.arange(0,len_total,10)])
        for ind_channel in range(16):
            plt.subplot(22,1,subplot_idx)
            subplot_idx += 1
            plt.plot(block_data['sev'][ind_channel,np.arange(0,len_total,100)])
            plt.title(channel_name + 'CH' + str(ind_channel))
    elif channel_name == 'path':
        subplot_idx -= 1
        pass
    else:
        plt.plot(block_data[channel_name])
        plt.title(channel_name)


plt.tight_layout() 
plt.draw()

embed()
