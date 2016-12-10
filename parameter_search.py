import pdb

import itertools
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import scipy

import supervised_learning

plt.ion()

def gsearch_expsel(features, targets, ids, 
        learner_params=None, theta_0=None, 
        iter_max=-1):
    """
    #TODO
    """
    iter_tot = 0
    grid_points = 7
    scores = []
    for theta_tople in itertools.product(np.linspace(0.001, 1.0, grid_points),
            repeat=3):
        iter_tot += 1
        print('gsearch: ' + str(100*iter_tot/grid_points**3) + ' %')
        theta = np.vstack(theta_tople)
        theta /= np.sum(theta)

        # transform to get a new target
        targets_trans = targets.dot(theta)
        # train and evaluate an estimator using the new target
        accu = supervised_learning.regress_svr(features=features, 
            targets=targets_trans.ravel(), ids=ids, 
            params={'kernel':['linear'], 
                'C':10.0**np.linspace(0,2,10)}, n_folds=10)
        # store score and theta pair
        scores.append((accu, theta))
    
    points = np.vstack([np.hstack(t[1]) for t in scores])
    colors = np.vstack([t[0]['r_value'] for t in scores])
    ind_best = np.argsort(colors, axis=0)[::-1][0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, marker='o')
    ax.scatter(points[ind_best,0], points[ind_best,1], points[ind_best,2], 
            c=colors[ind_best], marker='*', s=160)
    
    
    embed()
    return scores


def lsearch_expsel(features, targets, ids, 
        learner_params=None, theta_0=None, 
        iter_max=-1):
    """
    #TODO
    """
    # calc the initial value
    if theta_0 is None:
        theta_0 = (1.0/features.shape[1]) * np.ones((features.shape[1],1), dtype=np.float)

    theta = theta_0
    
    iter_tot = 1
    while iter_tot<iter_max or iter_max==-1:
        print('Local param search iter <'+str(iter_tot)+'> :')
        iter_tot += 1
        
        # get successors
        successors = find_successors(theta, delta=1.0e-2)

        # for each successor
        scores = []
        for succ in successors:
            # transform to get a new target
            targets_trans = targets.dot(succ)

            # train and evaluate an estimator using the new target
            accu = supervised_learning.regress_svr(features=features, 
                targets=targets_trans.ravel(), ids=ids, 
                params={'kernel':['linear'], 
                    'C':10.0**np.linspace(0,2,10)}, n_folds=10)
            # store score and theta_succ pair
            scores.append((accu, succ))
        
        # pick the best score theta, and update
        scores.sort(key=lambda k: k[0]['MAE'])
        theta = scores[0][1]
        #embed()
        # check termination conditions
        print('Theta optimum is: '+str(theta.flatten().tolist()))
        print('    score = '+str(scores[0][0]['MAE']))

    return theta

def find_successors(theta, delta):
    successors = []

    for ind in range(theta.shape[0]):
        # pos delta
        theta_succ = theta.copy()
        theta_succ[ind,0] += delta
        theta_succ /= np.sum(theta_succ)
        successors.append(theta_succ)
        # neg delta
        theta_succ = theta.copy()
        theta_succ[ind,0] -= delta
        theta_succ /= np.sum(theta_succ)
        successors.append(theta_succ)

    return successors


if __name__ == "__main__":
    succ = find_successors((1.0/3.0)*np.ones((3,1), dtype=np.float))
    embed()

