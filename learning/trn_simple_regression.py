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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import learning.nn_simple_regression as nn


# parameters
N_FE = 3
SIZE_BATCH = 72
SIZE_HIDDENS = [5]
RATE_LEARNING = 0.0001
MAX_STEPS = 10000
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
  targets_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
  return features_placeholder, targets_placeholder


def fill_feed_dict(dataset, features_pl, targets_pl):
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
  targets_feed = dataset[1]
  feed_dict = {
      features_pl: features_feed,
      targets_pl: targets_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_mae,
            features_placeholder,
            targets_placeholder,
            dataset):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_mae: The Tensor that returns mae of predictions.
    features_placeholder: The features placeholder.
    targets_placeholder: The targets placeholder.
    dataset: The set of images and labels to evaluate.
  """
  """
  # And run one epoch of eval. 
  mae = 0
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  """
  print('Evaluating ...')
  # TODO: add eval!

def run_training(dataset_trn,dataset_val=None,dataset_tst=None):
  """Train nn for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    features_placeholder, targets_placeholder = placeholder_inputs(SIZE_BATCH)

    # Build a Graph that computes predictions from the inference model.
    preds = nn.inference(features_placeholder,
                             SIZE_HIDDENS)

    # Add to the Graph the Ops for loss calculation.
    loss = nn.loss(preds, targets_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = nn.training(loss, RATE_LEARNING)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_model = nn.evaluation(preds, targets_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(DIR_LOG, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(MAX_STEPS):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(dataset_trn,
                                 features_placeholder,
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
        print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
        checkpoint_file = os.path.join(DIR_LOG, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        feed_dict = fill_feed_dict(dataset_trn, 
                features_placeholder, targets_placeholder)
        accu = sess.run(eval_model, feed_dict=feed_dict)
        print(accu)
        """
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_mae,
                features_placeholder,
                targets_placeholder,
                dataset_val)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_mae,
                features_placeholder,
                targets_placeholder,
                dataset_tst)
        """

