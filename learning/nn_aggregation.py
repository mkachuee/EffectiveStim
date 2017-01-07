"""Builds a simple regression network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pdb

import tensorflow as tf


# define input dimensions
N_FE = 3
N_SN = 3

def set_params(params=None):
    global N_FE, N_SN
    if params != None:
        N_FE = params['N_FE']
        N_SN = params['N_SN']

def inference(features, scores_normalized, hidden_units, hidden_units_agg):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([N_FE, hidden_units[0]],
                            stddev=1.0 / math.sqrt(float(N_FE))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units[0]]),
                         name='biases')
    hidden = tf.nn.sigmoid(tf.matmul(features, weights) + biases)
  
  cnt_hid=1
  for unit_hid in hidden_units[1:]:
    cnt_hid += 1
    with tf.name_scope('hidden'+str(cnt_hid)):
        weights = tf.Variable(
                tf.truncated_normal([int(hidden.get_shape()[1]), unit_hid],
                            stddev=1.0 / math.sqrt(float(int(hidden.get_shape()[1])))),
                name='weights')
        biases = tf.Variable(tf.zeros([unit_hid]),
                         name='biases')
        hidden = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)

  
  # output
  with tf.name_scope('output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units[-1], 1],
                            stddev=1.0 / math.sqrt(float(int(hidden_units[-1])))),
        name='weights')
    biases = tf.Variable(tf.zeros([1]),
                         name='biases')
    preds = tf.matmul(hidden, weights) + biases
      
  # agg_Hidden_agg 1
  with tf.name_scope('agg_hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([N_SN, hidden_units_agg[0]],
                            stddev=1.0 / math.sqrt(float(N_SN))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units_agg[0]]),
                         name='biases')
    hidden = tf.nn.sigmoid(tf.matmul(scores_normalized, weights) + biases)
  
  cnt_hid=1
  for unit_hid in hidden_units_agg[1:]:
    cnt_hid += 1
    with tf.name_scope('agg_hidden'+str(cnt_hid)):
        weights = tf.Variable(
                tf.truncated_normal([int(hidden.get_shape()[1]), unit_hid],
                            stddev=1.0 / math.sqrt(float(int(hidden.get_shape()[1])))),
                name='weights')
        biases = tf.Variable(tf.zeros([unit_hid]),
                         name='biases')
        hidden = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)

  
  # agg output
  with tf.name_scope('agg_output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units_agg[-1], N_SN],
                            stddev=1.0 / math.sqrt(float(hidden_units_agg[-1]))),
        name='weights')


    biases = tf.Variable(tf.zeros([1]), name='biases')

    agg_preds =  tf.nn.softmax(tf.matmul(hidden, weights) + biases)
    #agg_preds_n =  (tf.matmul(hidden, weights) + biases)
    #agg_preds = tf.div(agg_preds_n,tf.reshape(
    #    tf.reduce_sum(agg_preds_n, axis=1), (-1,1)))
  
  #agg_preds_mean = tf.ones([tf.shape(preds)[0], 3], dtype=tf.float32) / 3.0
  #t_tmp = tf.ones([tf.shape(preds)[0], 1], dtype=tf.float32)
  #agg_preds_0 = tf.reshape(tf.stack([t_tmp*0.7,t_tmp*0.2,t_tmp*0.1], axis=1),
  #        (-1,3))

  return preds, agg_preds

def loss(preds, agg_preds, targets):
  """
  Calculates the loss.
  """
  agg_targets = tf.reshape(tf.reduce_sum(targets * agg_preds, axis=1),(-1,1))
  cost_mse = (agg_targets - preds) ** 2
  loss = tf.reduce_mean(cost_mse, name='agg_cost_mse')
  return loss

def training(loss, learning_rate, learning_rate_agg):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # create variable lists
  var_list = []
  var_list_agg = []
  for tval in tf.trainable_variables():
    tf.summary.histogram(tval.name, tval)
    if tval.name[:3] == 'agg':
      var_list_agg.append(tval)
    else:
      var_list.append(tval)
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer_agg = tf.train.GradientDescentOptimizer(learning_rate_agg)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step, 
          var_list=var_list)
  train_op_agg = optimizer_agg.minimize(loss, global_step=global_step, 
          var_list=var_list_agg)
  return tf.group(train_op, train_op_agg)


def evaluation(preds, agg_preds, targets):
  """
  Calculates mse
  """
  agg_targets = tf.reshape(tf.reduce_sum(targets * agg_preds, axis=1),(-1,1))
  err = agg_targets - preds 
  mae = tf.reduce_mean(tf.abs(err))
  std = tf.sqrt(tf.reduce_mean((err-tf.reduce_mean(err)) ** 2))
  # r_value calculation
  mu_t = tf.reduce_mean(agg_targets)
  mu_p = tf.reduce_mean(preds)
  nstd_t = tf.sqrt(tf.reduce_sum((agg_targets-mu_t) **2))
  nstd_p = tf.sqrt(tf.reduce_sum((preds-mu_p) **2))
  r_value = tf.reduce_sum((agg_targets-mu_t)*(preds-mu_p)) / (nstd_t*nstd_p)
  return {'MAE':mae, 'STD':std, 'R':r_value}
