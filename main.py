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

"""Main program using the SVM class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split
import svm


def parse_args():
    parser = argparse.ArgumentParser(
        description='SVM built using TensorFlow, for Wisconsin Breast Cancer Diagnostic Dataset')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='path where to save the TensorBoard logs')
    arguments = parser.parse_args()
    return arguments


def main(arguments):

    features = datasets.load_breast_cancer().data
    labels = datasets.load_breast_cancer().target

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.30, stratify=labels)

    model = svm.Svm(train_data=[train_features, train_labels], validation_data=[test_features, test_labels],
                    log_path=arguments.log_path)

    # model.train()



if __name__ == '__main__':
    args = parse_args()

    main(args)
