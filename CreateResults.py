
import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from IPython import embed

plt.ion()
#font = {'family' : 'times',
#        'weight' : 'bold',
#        'size'   : 22}
#
#matplotlib.rc('font', **font)
font_params = {'fontsize':16}
    #{'fontname':'Times New Roman', 'fontsize':16}#, 
    #'fontweight':'bold'}

data = scipy.io.loadmat('./run_outputs/results.mat')
data_mean = scipy.io.loadmat('./run_outputs/results_mean.mat')

err = data['test_targets'] - data['test_predictions']
err = err.transpose() * 100.0
err_mean = data_mean['test_targets'] - data_mean['test_predictions']
err_mean = err_mean.transpose() * 100.0

plt.figure()
plt.hist(err, 10, normed=0, facecolor='k', alpha=0.75, rwidth=0.9)
plt.xlabel('Error (%)', **font_params)
plt.ylabel('Frequency', **font_params)
plt.figure()
plt.hist(err_mean, 10, normed=0, facecolor='k', alpha=0.75, rwidth=0.9)
plt.xlabel('Error (%)', **font_params)
plt.ylabel('Frequency', **font_params)

plt.draw()
embed()
