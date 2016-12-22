# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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


def inference(features, hidden_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  #TODO: add support for multiple hidden layers
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([N_FE, hidden_units[0]],
                            stddev=1.0 / math.sqrt(float(N_FE))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden_units[0]]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(features, weights) + biases)
  # output
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units[0], 1],
                            stddev=1.0 / math.sqrt(float(hidden_units[0]))),
        name='weights')
    biases = tf.Variable(tf.zeros([1]),
                         name='biases')
    preds = tf.matmul(hidden1, weights) + biases
    
  return preds

def loss(preds, targets):
  """
  Calculates the loss.
  """
  cost_mse = (targets-preds)**2
  loss = tf.reduce_mean(cost_mse, name='cost_mse')
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


def evaluation(preds, targets):
  """
  Calculates mse
  """
  err = targets - preds
  mae = tf.reduce_mean(tf.abs(err))
  std = tf.sqrt(tf.reduce_mean((err-tf.reduce_mean(err)) ** 2))
  # r_value calculation
  mu_t = tf.reduce_mean(targets)
  mu_p = tf.reduce_mean(preds)
  nstd_t = tf.sqrt(tf.reduce_sum((targets-mu_t) **2))
  nstd_p = tf.sqrt(tf.reduce_sum((preds-mu_p) **2))
  r_value = tf.reduce_sum((targets-mu_t)*(preds-mu_p)) / (nstd_t*nstd_p)
  return {'MAE':mae, 'STD':std, 'R':r_value}
