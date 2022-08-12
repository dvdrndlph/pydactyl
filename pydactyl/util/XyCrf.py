__author__ = 'David Randolph'
# Copyright (c) 2021, 2022 David A. Randolph.
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


class XyCrf(ABC):
    def __init__(self, tags: list):
        self.feature_functions = []
        self.tags = tags
        self.tags.append('START')
        self.tags.append('STOP')
        self.m = len(tags)
        self.J = 0  # Number of global feature functions.
        self.weights = []

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

    def get_g_list(self, X, inference=True):
        g_list = list()
        for i in range(len(X)):
            if inference:
                g_i = self.get_g_i_dict(X, i)
                g_list.append(g_i)
        return g_list

    def add_feature_function(self, func):
        self.feature_functions.append(func)
        self.J += 1
        self.weights.append(0.0)

    def viterbi(self, X, g_list):
        # Modeled after Seong-Jin Kim's implementation.
        time_len = len(X)
        max_table = np.zeros((time_len, self.m))
        argmax_table = np.zeros((time_len, self.m), dtype='int64')

        tag_index_for_name = dict()
        tag_name_for_index = dict()
        tag_index = 0
        for tag_name in self.tags:
            tag_index_for_name[tag_name] = tag_index
            tag_name_for_index[tag_index] = tag_name
            tag_index += 1

        t = 0
        for tag_index in tag_index_for_name:
            max_table[t, tag_index] = g_list[t][('START', tag_name_for_index[tag_index])]

        for t in range(1, time_len):
            for tag_index in range(1, self.m):
                tag = tag_name_for_index[tag_index]
                max_value = -float('inf')
                max_tag_index = None
                for prev_tag_index in range(1, self.m):
                    prev_tag = tag_name_for_index[prev_tag_index]
                    value = max_table[t - 1, prev_tag_index] * g_list[t][(prev_tag, tag)]
                    if value > max_value:
                        max_value = value
                        max_tag_index = prev_tag_index
                max_table[t, tag_index] = max_value
                argmax_table[t, tag_index] = max_tag_index

        sequence = list()
        next_tag_index = max_table[time_len - 1].argmax()
        sequence.append(tag_name_for_index[next_tag_index])
        for t in range(time_len - 1, -1, -1):
            next_tag_index = argmax_table[t, next_tag_index]
            sequence.append(tag_name_for_index[next_tag_index])
        # return [self.label_dic[label_id] for label_id in sequence[::-1][1:]]
        return sequence

    def infer(self, X):
        g_list = self.get_g_list(X, inference=True)
        y_hat = self.viterbi(X, g_list)
        return y_hat


if __name__ == '__main__':
    tags = ['>1', '>2', '>3', '>4', '>5']
    xyc = XyCrf(tags=tags)