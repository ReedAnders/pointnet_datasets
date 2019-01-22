# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/guide/reading_data#reading_from_files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

import model


# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def decode(serialized_example):
    """Parses data, label, and sample weight from the given `serialized_example`."""
    
    features = tf.parse_single_example(
    serialized_example,
    
    # Defaults are not specified since both keys are required.
    features={
        'pointclouds_pl': tf.FixedLenFeature([], tf.string),
        'labels_pl': tf.FixedLenFeature([], tf.string),
        'smpws_pl': tf.FixedLenFeature([], tf.string),
    })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    data = tf.decode_raw(features['pointclouds_pl'], tf.float32)
    labels = tf.decode_raw(features['labels_pl'], tf.int32)
    sample_weights = tf.decode_raw(features['smpws_pl'], tf.float32)
    # data.set_shape((mnist.IMAGE_PIXELS))

    data = tf.reshape(data, [FLAGS.num_classes, FLAGS.num_points, 3])
    labels = tf.reshape(labels, [FLAGS.num_classes, FLAGS.num_points])
    sample_weights = tf.reshape(sample_weights, [FLAGS.num_classes, FLAGS.num_points])
    
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    # label = tf.cast(features['label'], tf.int32)

    return data, labels, sample_weights


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).

    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
  """
  if not num_epochs:
    num_epochs = None
  filename = os.path.join(FLAGS.train_dir, TRAIN_FILE
                          if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    dataset = dataset.map(decode)
    #import pdb; pdb.set_trace()
    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    dataset = dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.repeat(num_epochs)
    #dataset = dataset.batch(batch_size)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

  return iterator.get_next()


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

  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op


def run_training():
  """Train MNIST for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    pointcloud_batch, label_batch, sample_weight_batch = inputs(
        train=FLAGS.is_training, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    is_training = tf.constant(FLAGS.is_training, dtype=tf.bool)
    # Build a Graph that computes predictions from the inference model.
    logits, _ = model.get_model(pointcloud_batch, is_training, FLAGS.num_classes)

    # Add to the Graph the loss calculation.
    loss = model.get_loss(logits, label_batch, sample_weight_batch)

    # Add to the Graph operations that train the model.
    train_op = training(loss, FLAGS.learning_rate)


    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    with tf.Session() as sess:
      # Initialize the variables (the trained variables and the
      # epoch counter).
      sess.run(init_op)
      try:
        step = 0
        while True:  # Train until OutOfRangeError
          start_time = time.time()

          # Run one step of the model.  The return values are
          # the activations from the `train_op` (which is
          # discarded) and the `loss` op.  To inspect the values
          # of your ops or variables, you may include them in
          # the list passed to sess.run() and the value tensors
          # will be returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss])

          duration = time.time() - start_time

          # Print an overview fairly often.
          if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                       duration))
          step += 1
      except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                          step))


def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.')
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=2,
      help='Number of epochs to run trainer.')
  parser.add_argument(
      '--is_training',
      type=bool,
      default=True,
      help='Toggles train or test within model.')
  parser.add_argument(
      '--num_classes',
      type=int,
      default=32,
      help='Number of classes to segment in data.')
  parser.add_argument(
      '--batch_size', 
      type=int, 
      default=100, 
      help='Batch size.')
  parser.add_argument(
      '--optimizer', 
      type=str, 
      default='adam', 
      help='Set optimizer as "adam" or "momentum".')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='data/tf_serialized',
      help='Directory with the training data.')
  parser.add_argument(
      '--num_points',
      type=int,
      default=8192,
      help='Number of points in input point cloud')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
