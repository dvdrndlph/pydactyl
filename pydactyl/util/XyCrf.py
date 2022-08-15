__author__ = 'David Randolph'
# Copyright (c) 2022 David A. Randolph.
# Copyright (c) 2015 Seong-Jin Kim.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from math import log

SCALING_THRESHOLD = 1e250

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0
GRADIENT = None


def _training_callback(params):
    global ITERATION_NUM
    global SUB_ITERATION_NUM
    global TOTAL_SUB_ITERATIONS
    ITERATION_NUM += 1
    TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
    SUB_ITERATION_NUM = 0


def _log_conditional_likelihood(params, *args):
    """
    Calculate likelihood and gradient
    """
    xycrf = args[0]
    xycrf.log_likelihood()


def _gradient(params, *args):
    return GRADIENT * -1


class XyCrf(ABC):
    def __init__(self, tags: list):
        self.squared_sigma = 10.0
        self.feature_functions = []
        self.tags = tags
        self.tags.append('START')
        self.tags.append('STOP')
        self.tag_count = len(tags)
        self.feature_count = 0  # Number of global feature functions.
        self.weights = []

        self.tag_index_for_name = dict()
        self.tag_name_for_index = dict()
        tag_index = 0
        for tag_name in self.tags:
            self.tag_index_for_name[tag_name] = tag_index
            self.tag_name_for_index[tag_index] = tag_name
            tag_index += 1

    def get_g_i(self, y_prev, y, X, i):
        j = 0
        value = 0
        for func in self.feature_functions:
            weight = self.weights[j]
            value += weight * func(y_prev, y, X, i)
        return value

    def get_g_i_dict(self, X, i):
        # Our matrix is a dictionary
        g_i_dict = dict()
        for y_prev in self.tags:
            for y in self.tags:
                if not (y_prev in ('START', 'STOP') and y in ('START', 'STOP')):
                    g_i_dict[(y_prev, y)] = self.get_g_i(y_prev, y, X, i)
        return g_i_dict

    def get_g_matrix_list(self, X):
        g_list = list()
        for i in range(len(X)):
            matrix = np.zeros((self.tag_count, self.tag_count))
            g_i = self.get_g_i_dict(X, i)
            for (y_prev, y) in g_i:
                y_prev_index = self.tag_index_for_name[y_prev]
                y_index = self.tag_index_for_name[y]
                matrix[y_prev_index, y_index] = g_i[(y_prev, y)]
            g_list.append(matrix)
        return g_list

    def get_inference_g_list(self, X):
        g_list = list()
        for i in range(len(X)):
            g_i = self.get_g_i_dict(X, i)
            g_list.append(g_i)
        return g_list

    def add_feature_function(self, func):
        self.feature_functions.append(func)
        self.feature_count += 1
        self.weights.append(0.0)

    def viterbi(self, X, g_list):
        # Modeled after Seong-Jin Kim's implementation.
        time_len = len(X)
        max_table = np.zeros((time_len, self.tag_count))
        argmax_table = np.zeros((time_len, self.tag_count), dtype='int64')

        t = 0
        for tag_index in self.tag_index_for_name:
            max_table[t, tag_index] = g_list[t][('START', self.tag_name_for_index[tag_index])]

        for t in range(1, time_len):
            for tag_index in range(1, self.tag_count):
                tag = self.tag_name_for_index[tag_index]
                max_value = -float('inf')
                max_tag_index = None
                for prev_tag_index in range(1, self.tag_count):
                    prev_tag = self.tag_name_for_index[prev_tag_index]
                    value = max_table[t - 1, prev_tag_index] * g_list[t][(prev_tag, tag)]
                    if value > max_value:
                        max_value = value
                        max_tag_index = prev_tag_index
                max_table[t, tag_index] = max_value
                argmax_table[t, tag_index] = max_tag_index

        sequence = list()
        next_tag_index = max_table[time_len - 1].argmax()
        sequence.append(self.tag_name_for_index[next_tag_index])
        for t in range(time_len - 1, -1, -1):
            next_tag_index = argmax_table[t, next_tag_index]
            sequence.append(self.tag_name_for_index[next_tag_index])
        # return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]
        return sequence

    def infer(self, X):
        g_list = self.get_inference_g_list(X)
        y_hat = self.viterbi(X, g_list)
        return y_hat

    def forward_backward(self):
        pass

    def log_conditional_likelihood(self):
        expected_scores = np.zeros(self.feature_count)
        sum_log_Z = 0


    def train(self, X, Y):
        print('* Squared sigma:', self.squared_sigma)
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        self.weights, log_likelihood, information = \
            fmin_l_bfgs_b(func=_log_conditional_likelihood, fprime=_gradient,
                          x0=np.zeros(self.feature_count), args=[self],
                          callback=_training_callback)
        # training_data, feature_set, training_feature_data, empirical_counts, label_dic, squared_sigma
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
        print('* Likelihood: %s' % str(log_likelihood))

if __name__ == '__main__':
    tags = ['>1', '>2', '>3', '>4', '>5']
    xyc = XyCrf(tags=tags)