
"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time
import pdb

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import scipy
import sklearn
import tensorflow as tf
from IPython import embed
import matplotlib.pyplot as plt

import learning.nn_aggregation as nn


# parameters
N_FE = 3
SIZE_BATCH = None
SIZE_HIDDENS = [4,4]
SIZE_HIDDENS_AGG = [3]
RATE_LEARNING = 0.01
MAX_STEPS = 100000
MAX_EARLYSTOP = 5
DIR_LOG = './logs'

os.system('rm -r '+DIR_LOG)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    features_placeholder: features placeholder.
    targets_placeholder: targets placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  features_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         N_FE))
  scores_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         3))

  targets_placeholder = tf.placeholder(tf.float32, shape=(batch_size,3))
  return features_placeholder, scores_placeholder, targets_placeholder


def fill_feed_dict(dataset, features_pl, scores_pl, targets_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    targets_pl: The targets placeholder, from placeholder_inputs().
    features_pl: The features placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  # TODO: add batch support
  features_feed = dataset[0]
  scores_feed = dataset[1] / np.max(dataset[1], axis=1).reshape(-1,1)
  #scores_feed = np.zeros(dataset[0].shape, float) - 1  dataset[1] / np.max(dataset[1], axis=1).reshape(-1,1)
  targets_feed = dataset[1]
  #targets_feed[:,0] = targets_feed.mean(1)
  feed_dict = {
      features_pl: features_feed,
      scores_pl: scores_feed,
      targets_pl: targets_feed,
  }
  return feed_dict

def run_training(dataset_trn,dataset_val=None,dataset_tst=None):
  """Train nn for a number of steps."""
  # if we don't have any val data, select it from trn
  if not dataset_val:
    portion_val = 0.20  
    n_trn = dataset_trn[0].shape[0]
    inds_val = np.random.choice(n_trn, int(n_trn*portion_val), replace=False)
    dataset_val = (dataset_trn[0][inds_val],dataset_trn[1][inds_val])
    mask = np.ones(dataset_trn[0].shape[0], dtype=bool)
    mask[inds_val]=False
    dataset_trn = (dataset_trn[0][mask,:],dataset_trn[1][mask,:])
  
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    features_placeholder, scores_placeholder, targets_placeholder = \
            placeholder_inputs(SIZE_BATCH)

    # Build a Graph that computes predictions from the inference model.
    preds, preds_agg = nn.inference(features_placeholder, scores_placeholder,
                             SIZE_HIDDENS, SIZE_HIDDENS_AGG)

    # Add to the Graph the Ops for loss calculation.
    loss = nn.loss(preds, preds_agg, targets_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = nn.training(loss, RATE_LEARNING)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_model = nn.evaluation(preds, preds_agg, targets_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    #summary_writer = tf.summary.FileWriter(DIR_LOG, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    max_early_stop = MAX_EARLYSTOP
    cnt_early_stop = 0
    previous_loss = 0.0
    for step in xrange(MAX_STEPS):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(dataset_trn,
                                 features_placeholder,
                                 scores_placeholder,
                                 targets_placeholder)
      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)
      duration = time.time() - start_time
      
      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('-'*40)
        print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        #summary_str = sess.run(summary, feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, step)
        #summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 100 == 0 or (step + 1) == MAX_STEPS:
        
        #checkpoint_file = os.path.join(DIR_LOG, 'model.ckpt')
        #saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        feed_dict = fill_feed_dict(dataset_trn, 
                features_placeholder, scores_placeholder, targets_placeholder)
        accu = sess.run(eval_model, feed_dict=feed_dict)
        print(accu)
        # Evaluate against the validation set.
        if dataset_val:
            print('Validation Data Eval:')
            feed_dict = fill_feed_dict(dataset_val, 
                    features_placeholder, scores_placeholder,
                    targets_placeholder)
            accu = sess.run(eval_model, feed_dict=feed_dict)
            print(accu)
            # update early stop and terminate, if necessary
            if accu['MAE'] > previous_loss:
                cnt_early_stop += 1
            else:
                cnt_early_stop = 0
            if cnt_early_stop > max_early_stop:
                break
            previous_loss = accu['MAE']
            print('Eearly Stop counter: '+\
                    str(cnt_early_stop)+'/'+str(max_early_stop))
        if dataset_tst:
            print('Test Data Eval:')
            feed_dict = fill_feed_dict(dataset_tst, 
                    features_placeholder, scores_placeholder,
                    targets_placeholder)
            accu = sess.run(eval_model, feed_dict=feed_dict)
            print(accu)
    # Evaluate against the test set.
    if dataset_tst:
        print('Test Data Eval:')
        feed_dict = fill_feed_dict(dataset_tst, 
                features_placeholder, scores_placeholder,
                targets_placeholder)
        accu = sess.run(eval_model, feed_dict=feed_dict)
        preds_tst = sess.run(preds, feed_dict=feed_dict)
        agg_preds = sess.run(preds_agg, feed_dict=feed_dict)
        
        targets_tst = np.sum(dataset_tst[1] * agg_preds, axis=1).reshape(-1,1)
        print(accu)
        #pdb.set_trace()
        return preds_tst, targets_tst

def regress_nn(features, targets, ids, params=None, 
        debug=False, n_folds=10, seed=None):
    """
    n-fold train and test.
    """
    # random permutation
    np.random.seed(seed)
    ind_perms = np.random.permutation(features.shape[0])
    features = features[ind_perms]
    targets = targets[ind_perms]
    ids = ids[ind_perms]
    # normalize features
    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    
    # to k fold test and traini
    test_ids = []
    test_targets = []
    test_predictions = []
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    for ind_train, ind_test in kf.split(targets):
        # create a cross validated model for each fold
        dataset_trn = (features[ind_train],targets[ind_train])
        dataset_tst = (features[ind_test],targets[ind_test])
        preds_test, targets_test = run_training(dataset_trn=dataset_trn, 
                dataset_tst=dataset_tst)
        
        test_predictions.append(preds_test)
        test_targets.append(targets_test)
        test_ids.append(ids[ind_test])
    # evaluate the model
    test_targets = np.vstack(test_targets)
    test_predictions = np.vstack(test_predictions)
    test_ids = np.vstack(test_ids)
    mae = np.around((np.abs(test_targets-test_predictions)).mean(), 
            decimals=4)
    std = np.around(np.std(test_targets-test_predictions), 
            decimals=4)
    mae_null = np.around((np.abs(test_targets-test_targets.mean())).mean(), 
            decimals=4)
    std_null = np.around(np.std(test_targets-test_targets.mean()), 
            decimals=4)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            test_targets.transpose(),test_predictions.transpose()) 
    if debug:
        print('MAE is: ' + str(mae))
        print('Null MAE is: ' + str(mae_null))
        print('STD is: ' + str(std))
        print('Null STD is: ' + str(std_null))
        print('r is: ' + str(r_value))
        #print('Null r is: ' + str(r_value_null))
        plt.figure()
        plt.plot(test_targets,test_predictions, 'o')
        reg_range = np.linspace(np.min(test_targets), 
                np.max(test_targets), 100)
        plt.plot(reg_range,reg_range, '.')
        plt.plot(reg_range, r_value*reg_range+intercept)
        plt.xlabel('Target')
        plt.ylabel('Prediction')
        plt.title('r = '+str(np.around(r_value, 2)) + \
                '\n MAE = '+str(100*mae) + ', STD= '+str(100*std))
        plt.axis('equal')
        embed()
    
    return {'r_value':r_value, 'MAE':mae, 'STD':std}
