# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Support Vector Machine using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import numpy as np
import sys
import tensorflow as tf

BATCH_SIZE = 50
NUM_CLASSES = 2


class SVM:

    def __init__(self, svm_c, num_epochs, log_path, num_features):
        self.svm_c = svm_c
        self.num_epochs = num_epochs
        self.log_path = log_path
        self.num_features = num_features

        def __graph__():
            """Building the inference graph"""

            with tf.name_scope('input'):

                # [BATCH_SIZE, 30]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_features], name='x_input')

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')

            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            with tf.name_scope('training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.get_variable(name='weights',
                                             initializer=tf.random_normal([self.num_features, 1], stddev=0.01))
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable(name='biases', initializer=tf.constant([0.1]))
                    self.variable_summaries(bias)
                with tf.name_scope('Wx_plus_b'):
                    output = tf.matmul(x_input, weight) + bias
                    tf.summary.histogram('pre-activations', output)

            with tf.name_scope('svm'):
                regularization = 0.5 * tf.reduce_sum(tf.square(weight))
                hinge_loss = tf.reduce_sum(
                    tf.square(tf.maximum(tf.zeros([BATCH_SIZE, NUM_CLASSES]), 1 - y_input * output)))
                with tf.name_scope('loss'):
                    loss = regularization + self.svm_c * hinge_loss
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = tf.sign(output)
                predicted_class = tf.identity(predicted_class, name='prediction')
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.learning_rate = learning_rate
            self.loss = loss
            self.optimizer = optimizer
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write('\n <log> Building graph...')
        __graph__()
        sys.stdout.write('</log> \n')

    def train(self, train_data):
        pass

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
