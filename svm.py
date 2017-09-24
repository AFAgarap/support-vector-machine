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


class SVM:

    def __init__(self, train_data, validation_data, log_path):
        self.train_data = train_data
        self.validation_data = validation_data
        self.log_path = log_path

        def __graph__():
            """Building the inference graph"""

            with tf.name_scope('input'):

                # [BATCH_SIZE, 30]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, 30], name='x_input')

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.float32, shape=[None], name='y_input')

            self.x_input = x_input
            self.y_input = y_input

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
