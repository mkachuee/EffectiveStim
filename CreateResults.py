
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from IPython import embed

plt.ion()

data = scipy.io.loadmat('./results.mat')
data_mean = scipy.io.loadmat('./results_mean.mat')

err = data['test_targets'] - data['test_predictions']
err = err.transpose()
err_mean = data_mean['test_targets'] - data_mean['test_predictions']
err_mean = err_mean.transpose()

plt.figure()
plt.hist(err, 10, normed=0, facecolor='k', alpha=0.75, rwidth=0.9)
plt.xlabel('Error (%)')
plt.ylabel('Frequency')
plt.figure()
plt.hist(err_mean, 10, normed=0, facecolor='k', alpha=0.75, rwidth=0.9)
plt.xlabel('Error (%)')
plt.ylabel('Frequency')

plt.draw()
embed()
