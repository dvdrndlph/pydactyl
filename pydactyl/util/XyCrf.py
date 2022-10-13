__author__ = 'David Randolph'
# Copyright (c) 2022 David A. Randolph.
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
#
import pprint
import random
from pathlib import Path
from pathos.multiprocessing import ProcessingPool, cpu_count
import dill
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from math import log, exp

START_TAG = '^'
STOP_TAG = '$'

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
    return xycrf.log_conditional_likelihood()


def _gradient(params, *args):
    xycrf = args[0]
    return xycrf.get_gradient()


class XyCrf:
    def __init__(self, optimize=True):
        self.testing = False
        self.optimize = optimize
        self.training_data = None
        self.feature_functions = []
        self.function_index_name = {}
        self.tag_count = 0
        self.tags = set()
        self.feature_count = 0  # Number of global feature functions.
        self.weights = []
        self.tag_index_for_name = {}
        self.tag_name_for_index = {}
        self.gradient = None

    def print_function_weights(self):
        weight_str = 'weights:'
        for index in range(self.feature_count):
            weight_str += f' {self.function_index_name[index]}: {self.weights[index]}'
        print(weight_str)

    def get_gradient(self):
        return self.gradient

    def set_gradient(self, new_values):
        if len(new_values) != self.feature_count:
            raise Exception("Bad gradient settings.")
        self.gradient = np.ndarray((self.feature_count,))
        for i in range(self.feature_count):
            self.gradient[i] = new_values[i]

    def set_tags(self, tag_set: set):
        self.tags = tag_set
        self.tags.add(START_TAG)
        self.tags.add(STOP_TAG)
        self.tag_count = len(self.tags)
        self.tag_index_for_name = {}
        self.tag_name_for_index = {}
        tag_index = 0
        for tag_name in self.tags:
            self.tag_index_for_name[tag_name] = tag_index
            self.tag_name_for_index[tag_index] = tag_name
            tag_index += 1

    def get_g_i(self, y_prev, y, x_bar, i):
        sum_of_weighted_features = 0
        for j in range(self.feature_count):
            weight = self.weights[j]
            func = self.feature_functions[j]
            func_name = self.function_index_name[j]
            feature_val = func(y_prev, y, x_bar, i)
            # if feature_val != 0 and self.testing:
            #     x_bar_str = "".join(list(itertools.chain(*x_bar)))
            #     print(f'{func_name}({y_prev},{y}, {x_bar_str}, {i}) returned {feature_val}')
            sum_of_weighted_features += weight * feature_val
        return sum_of_weighted_features

    def get_g_i_dict(self, x_bar, i):
        # Our matrix is a dictionary.
        g_i_dict = {}
        for y_prev in self.tags:
            for y in self.tags:
                g_i_dict[(y_prev, y)] = self.get_g_i(y_prev, y, x_bar, i)
        return g_i_dict

    def get_g_dicts(self, x_bar):
        g_dict_list = []
        for i in range(len(x_bar)):
            g_i = self.get_g_i_dict(x_bar, i)
            g_dict_list.append(g_i)
        return g_dict_list

    def add_feature_function(self, func, name=None):
        self.feature_functions.append(func)
        if name is None:
            name = "f_{}".format(self.feature_count)
        self.function_index_name[self.feature_count] = name
        self.feature_count += 1
        self.weights.append(0.0)

    def add_feature_functions(self, functions):
        for func in functions:
            self.add_feature_function(func=func)

    def clear_feature_functions(self):
        self.feature_functions = []
        self.function_index_name = {}
        self.feature_count = 0
        self.weights = []

    def init_weights(self):
        self.weights = []
        for j in range(len(self.feature_functions)):
            self.weights.append(0.0)

    def big_u(self, k, v_tag, g_dicts, memo):
        if (k, v_tag) in memo:
            return memo[(k, v_tag)]

        value = 0
        if k == 0:
            if v_tag == START_TAG:
                value = 1
            memo[(k, v_tag)] = value
            return value

        max_value = -1 * float('inf')
        for u_tag in self.tags:
            value = self.big_u(k-1, u_tag, g_dicts, memo)
            value += g_dicts[k][(u_tag, v_tag)]
            if value > max_value:
                max_value = value
        memo[(k, v_tag)] = max_value
        return max_value

    def viterbi(self, x_bar, g_dicts):
        n = len(x_bar)
        big_u_dict = {}
        memo = {}
        for k in range(0, n):
            for v_tag in self.tags:
                max_value = self.big_u(k=k, v_tag=v_tag, g_dicts=g_dicts, memo=memo)
                big_u_dict[(k, v_tag)] = max_value

        k_tag = None
        last_max = -1 * float('inf')
        for v_tag in sorted(self.tags):
            tag_max = big_u_dict[(n-1, v_tag)]
            if tag_max > last_max:
                last_max = tag_max
                k_tag = v_tag
        y_hat = [k_tag]
        for k in range(n-1, 0, -1):
            max_value = -1 * float('inf')
            max_tag = None
            for u_tag in sorted(self.tags):
                value = big_u_dict[(k-1, u_tag)]
                value += g_dicts[k][(u_tag, k_tag)]
                if value > max_value:
                    max_value = value
                    max_tag = u_tag
            y_hat.insert(0, max_tag)
            k_tag = max_tag

        return y_hat

    def alpha(self, k_plus_1, v_tag, g_dicts, memo):
        if (k_plus_1, v_tag) in memo:
            return memo[(k_plus_1, v_tag)]

        sum_total = 0
        if k_plus_1 == 0:
            if v_tag == START_TAG:
                sum_total = 1
            memo[(k_plus_1, v_tag)] = sum_total
            return sum_total

        k = k_plus_1 - 1
        for u_tag in self.tags:
            g = g_dicts[k_plus_1][(u_tag, v_tag)]
            exp_g = exp(g)
            alphie = self.alpha(k, u_tag, g_dicts, memo)
            sum_total += alphie * exp_g

        memo[(k_plus_1, v_tag)] = sum_total
        return sum_total

    def beta(self, u_tag, k, g_dicts, memo={}):
        if (u_tag, k) in memo:
            return memo[(u_tag, k)]

        sum_total = 0
        n = len(g_dicts)  # Length of the sequence
        if k == n - 1:
            if u_tag == STOP_TAG:
                sum_total = 1
            memo[(u_tag, k)] = sum_total
            return sum_total

        for v_tag in self.tags:
            g = g_dicts[k+1][(u_tag, v_tag)]
            exp_g = exp(g)
            betty = self.beta(v_tag, k+1, g_dicts, memo)
            sum_total += exp_g * betty

        memo[(u_tag, k)] = sum_total
        return sum_total

    def big_z_forward(self, g_dicts):
        n = len(g_dicts)
        big_z = self.alpha(n-1, STOP_TAG, g_dicts, memo={})
        return big_z

    def big_z_backward(self, g_dicts):
        big_z = self.beta(START_TAG, 0, g_dicts, memo={})
        return big_z

    def expectation_for_function(self, function_index, x_bar, g_dicts, validate=False):
        func = self.feature_functions[function_index]
        func_name = self.function_index_name[function_index]
        n = len(x_bar)
        big_z = self.big_z_forward(g_dicts)
        if big_z == 0:
            raise Exception('Z cannot be 0.')  # But it could be, couldn't it?
        if validate:
            # Forward and backward Z values are very close, but not identical.
            big_z_beta = self.big_z_backward(g_dicts)
            if abs(big_z - big_z_beta) / big_z > 0.01:
                raise Exception(f"Z values do not match: {big_z} vs. {big_z_beta}")

        expectation = 0.0
        for i in range(1, n):
            for y_prev in self.tags:
                for y in self.tags:
                    feature_value = func(y_prev=y_prev, y=y, x_bar=x_bar, i=i)
                    if feature_value == 0:
                        continue
                    alpha_value = self.alpha(k_plus_1=i-1, v_tag=y_prev, g_dicts=g_dicts, memo={})
                    beta_value = self.beta(u_tag=y, k=i, g_dicts=g_dicts, memo={})
                    g_i_value = g_dicts[i][(y_prev, y)]
                    try:
                        exp_g_i_value = exp(g_i_value)
                    except:
                        raise Exception(f'Exponentiated big number go boom for {func_name}.')
                    expectation += feature_value * ((alpha_value * exp_g_i_value * beta_value) / big_z)
        return expectation, big_z

    def infer(self, x_bar):
        self.testing = True
        g_dicts = self.get_g_dicts(x_bar)
        y_hat = self.viterbi(x_bar, g_dicts=g_dicts)
        return y_hat

    def infer_all(self, data):
        y_hats = []
        for example in data:
            x_bar = example[0]
            y_hat = self.infer(x_bar)
            y_hats.append(y_hat)
        return y_hats

    def big_f(self, function_index, x_bar, y_bar):
        n = len(y_bar)
        func = self.feature_functions[function_index]
        # func_name = self.function_index_name[function_index]
        sum_total = 0
        for i in range(1, n):
            token = x_bar[i][0]
            val = func(y_prev=y_bar[i-1], y=y_bar[i], x_bar=x_bar, i=i)
            sum_total += val
        # print("Returning")
        return sum_total

    def learn_from_function(self, function_index, x_bar, y_bar, g_dicts):
        actual_val = self.big_f(function_index=function_index, x_bar=x_bar, y_bar=y_bar)
        expected_val, example_big_z = self.expectation_for_function(g_dicts=g_dicts,
                                                                    function_index=function_index, x_bar=x_bar)
        return actual_val, expected_val, example_big_z

    def learn_from_functions(self, x_bar, y_bar, g_dicts):
        pool = ProcessingPool(nodes=cpu_count())
        x_bars = []
        y_bars = []
        for _ in range(self.feature_count):
            x_bars.append(x_bar)
            y_bars.append(y_bar)
        results = pool.map(_learn_from_function, zip([self]*self.feature_count,
                                                     [x for x in range(self.feature_count)],
                                                     x_bars, y_bars,
                                                     [g_dicts]*self.feature_count))
        return results

    def gradient_for_all_training(self):
        function_count = len(self.feature_functions)
        gradient = []
        big_z = 0
        for example in self.training_data:
            x_bar = example[0]
            y_bar = example[1]
            g_dicts = self.get_g_dicts(x_bar=x_bar)
            for j in range(function_count):
                # FIXME: Run next command in parallel for all j.
                actual_val, expected_val, example_big_z = self.learn_from_function(g_dicts=g_dicts,
                                                                                   function_index=j,
                                                                                   x_bar=x_bar, y_bar=y_bar)
                if len(gradient) == j:
                    gradient.append(0)
                gradient[j] += actual_val - expected_val
                big_z += example_big_z
        return gradient, big_z

    def log_conditional_likelihood(self):
        gradient, big_z = self.gradient_for_all_training()
        pprint.pprint(gradient)
        weighted_feature_sum = 0
        function_count = len(self.feature_functions)
        for j in range(function_count):
            global_feature_val = 0
            for example in self.training_data:
                x_bar = example[0]
                y_bar = example[1]
                global_feature_val += self.big_f(function_index=j, x_bar=x_bar, y_bar=y_bar)
            weighted_feature_sum += self.weights[j] * global_feature_val
        likelihood = weighted_feature_sum - log(big_z)
        self.set_gradient(gradient)
        return likelihood

    def stochastic_gradient_ascent_train(self, regularization=0.0005,
                                         learning_rate=0.01, attenuation=1, epochs=1,
                                         seeder=lambda: 0.27):
        function_count = len(self.feature_functions)
        self.init_weights()
        # FIXME: There must be a stopping condition other than the last training example.
        # Elkan gives us a rule of thumb that says 3-100 epochs, but with how expensive
        # the "edge-observation" functions of both x and y values are, we need to run lean.

        block_size = 1000
        epoch_number = 1
        for epoch in range(epochs):
            print(f'Starting pass number {epoch_number} (of {epochs}) through the training set.')
            example_num = 0
            if seeder is not None:
                random.shuffle(self.training_data, seeder)
            for example in self.training_data:
                # global_feature_vals = np.zeros(self.feature_count)
                # expected_vals = np.zeros(self.feature_count)
                x_bar = example[0]
                y_bar = example[1]
                g_dicts = self.get_g_dicts(x_bar=x_bar)
                learnings = []
                if self.optimize:
                    learnings = self.learn_from_functions(x_bar=x_bar, y_bar=y_bar, g_dicts=g_dicts)
                else:
                    for j in range(function_count):
                        global_feature_val, expected_val, _ = self.learn_from_function(g_dicts=g_dicts,
                                                                                       function_index=j,
                                                                                       x_bar=x_bar, y_bar=y_bar)
                        learnings.append([global_feature_val, expected_val])
                for j in range(function_count):
                    global_feature_val = learnings[j][0]
                    expected_val = learnings[j][1]
                    reg_val = 2 * regularization * self.weights[j]
                    self.weights[j] = self.weights[j] + learning_rate * ((global_feature_val - expected_val) - reg_val)

                example_num += 1
                if example_num % block_size == 0:
                    print(f'Example {epoch_number}:{example_num} processed with learning rate {learning_rate}.')
                    self.print_function_weights()
            learning_rate *= attenuation
            epoch_number += 1
        print("The stochastic gradient has been ascended.")

    def train(self):
        self.init_weights()
        print('* Start L-BGFS')
        print('   ========================')
        print('   iter(sit): likelihood')
        print('   ------------------------')
        self.weights, log_likelihood, information = \
            fmin_l_bfgs_b(func=_log_conditional_likelihood, fprime=_gradient,
                          x0=np.zeros(self.feature_count), args=[self],
                          callback=_training_callback)
        print('   ========================')
        print('   (iter: iteration, sit: sub iteration)')
        print('* Training has been finished with %d iterations' % information['nit'])

        if information['warnflag'] != 0:
            print('* Warning (code: %d)' % information['warnflag'])
            if 'task' in information.keys():
                print('* Reason: %s' % (information['task']))
            print('* Likelihood: %s' % str(log_likelihood))

    def pickle(self, path_str):
        print("Pickling to path {}.".format(path_str))
        pickle_fh = open(path_str, 'wb')
        # dill.dump(self, pickle_fh, protocol=pickle.HIGHEST_PROTOCOL)
        dill.dump(self, pickle_fh, protocol=dill.HIGHEST_PROTOCOL)
        pickle_fh.close()

    @staticmethod
    def unpickle(path_str):
        path = Path(path_str)
        if path.is_file():
            pickle_fh = open(path_str, 'rb')
            unpickled_obj = dill.load(pickle_fh)
            pickle_fh.close()
            print("Unpickled object from path {}.".format(path_str))
            return unpickled_obj
        return None


def _learn_from_function(arg, **kwarg):
    return XyCrf.learn_from_function(*arg, **kwarg)


if __name__ == '__main__':
    rh_tags = ['>1', '>2', '>3', '>4', '>5']
    xyc = XyCrf()
    xyc.set_tags(tag_set=set(rh_tags))
