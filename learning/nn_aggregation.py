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
                            stddev=1.0 / math.sqrt(float(N_FE))),
                name='weights')
        biases = tf.Variable(tf.zeros([unit_hid]),
                         name='biases')
        hidden = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)

  
  # output
  with tf.name_scope('output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units[-1], 1],
                            stddev=1.0 / math.sqrt(float(hidden_units[-1]))),
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
                            stddev=1.0 / math.sqrt(float(N_SN))),
                name='weights')
        biases = tf.Variable(tf.zeros([unit_hid]),
                         name='biases')
        hidden = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)

  
  # agg output
  with tf.name_scope('agg_output'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units_agg[-1], 3],
                            stddev=1.0 / math.sqrt(float(hidden_units_agg[-1]))),
        name='weights', trainable=False)

    #weights_tmp = tf.zeros([hidden_units_agg[-1], 3])

    biases = tf.Variable(tf.zeros([1])+0.33,
                         name='biases', trainable=False)

    #biases_tmp = tf.zeros([1])+0.33
    agg_preds =  tf.nn.softmax(tf.matmul(hidden, weights) + biases)

  
  #agg_preds_mean = tf.ones([tf.shape(preds)[0], 3], dtype=tf.float32) / 3.0

  return preds, agg_preds

def loss(preds, agg_preds, targets):
  """
  Calculates the loss.
  """
  #agg_targets = tf.diag_part(tf.matmul(targets, tf.transpose(agg_preds)))
  #cost_mse = (agg_targets-preds)**2
  
  #cost_mse = (tf.matmul(targets,tf.constant([1.0,0.0,0.0],shape=(3,1))) -preds)**2
  #pdb.set_trace()
  #agg_targets = tf.reduce_sum((tf.matmul(targets,tf.transpose(agg_preds))) \
  #  * tf.eye(tf.shape(targets)[0]), axis=1)
  #agg_targets = tf.diag_part(tf.matmul(targets,tf.transpose(agg_preds)))
  #agg_targets = tf.reduce_sum(targets * agg_preds, axis=1)
  agg_targets = tf.reshape(tf.reduce_sum(targets * agg_preds, axis=1),(-1,1))
  #pdb.set_trace()
  cost_mse = (agg_targets - preds) ** 2
  loss = tf.reduce_mean(cost_mse, name='agg_cost_mse')
  return loss

def training(loss, learning_rate):
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
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(preds, agg_preds, targets):
  """
  Calculates mse
  """
  #return {} # FIXME
  #agg_targets = tf.diag_part(tf.matmul(targets, tf.transpose(agg_preds)))
  #agg_targets = tf.matmul(targets,tf.constant([1.0,0.0,0.0],shape=(3,1)))
  #agg_targets = tf.diag_part(tf.matmul(targets,tf.transpose(agg_preds)))
  #agg_targets = tf.reduce_sum((tf.matmul(targets,tf.transpose(agg_preds))) \
  #  * tf.eye(tf.shape(targets)[0]), axis=1)
  #agg_targets = tf.reduce_sum(targets * agg_preds, axis=1)
  agg_targets = tf.reshape(tf.reduce_sum(targets * agg_preds, axis=1),(-1,1))
  #agg_targets = tf.matmul(targets,tf.transpose(agg_preds))
  #tf.matmul(targets,tf.constant([1.0, 0.0, 0.0])) #[:,0]
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
