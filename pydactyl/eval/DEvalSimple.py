__author__ = 'David Randolph'
# Copyright (c) 2014-2018 David A. Randolph.
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
from krippendorff import alpha
import re
import random
import numpy as np
# import matplotlib.pyplot as plt
# import sklearn.cluster as cluster
# import seaborn as sns
# import time
import math
from pydactyl.dcorpus.DCorpus import DCorpus
# from pydactyl.dactyler import Constant
# from pydactyl.dcorpus.DNote import DNote
from pydactyl.dcorpus.DAnnotation import DAnnotation

class DEvalSimple(ABC):
    def __init__(self):
        self._dactyler = None
        self._d_corpus = DCorpus()
        self._gold = dict()
        self._gold['upper'] = list()
        self._gold['lower'] = list()

    @staticmethod
    def _max_distance_for_method(method):
        if method == "natural":
            # d_max = Constant.MAX_NATURAL_EDIT_DISTANCE
            # FIXME: desegregated max turns DCPG murky and useless.
            d_max = 4
        elif method == "hamming":
            d_max = 1
        else:
            raise Exception("Not ready to measure {} gain".format(method))
        return d_max

    @staticmethod
    def _proximal_gain(test_abcdf, gold_handed_strikers, gold_vote_counts, staff, method="hamming"):
        testee = DAnnotation.abcdf_to_handed_strike_digits(test_abcdf, staff=staff)
        d_max = DEvalSimple._max_distance_for_method(method)
        nugget_index = 0
        total_gain = 0
        for nugget in gold_handed_strikers:
            nugget_gain = 0
            for i in range(len(testee)):
                distance = DAnnotation.strike_distance_cost(gold_handed_digit=nugget[i],
                                                            test_handed_digit=testee[i],
                                                            method=method)
                nugget_gain += (d_max - distance)
            nugget_gain *= gold_vote_counts[nugget_index]
            nugget_index += 1
            total_gain += nugget_gain
        return total_gain

    @abstractmethod
    def load_data(self):
        return

    def gold_asts(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)
        asts = list()
        for abcdf in score_gold:
            ast = DAnnotation.abcdf_to_ast(abcdf)
            asts.append(ast)
        return asts

    def gold_list_of_handed_strike_lists(self, score_index, staff="upper", last_digit=None):
        gold_asts = self.gold_asts(score_index, staff=staff, last_digit=last_digit)
        list_of_strike_lists = list()
        for gold_ast in gold_asts:
            nuggets = DAnnotation.ast_to_handed_strike_digits(gold_ast, staff=staff)
            list_of_strike_lists.append(nuggets)
        return list_of_strike_lists

    def gold_clusters(self, score_index, staff="upper", last_digit=None):
        score_gold = self._score_gold(score_index=score_index, staff=staff, last_digit=last_digit)

    @staticmethod
    def krippendorfs_alpha(upper_rh_advice, exercise_upper_gold):
        fingerings = list(upper_rh_advice)
        fingerings.pop(0)
        finger_ints = list(map(int, fingerings))
        exercise_upper_gold.append(finger_ints)
        krip = alpha(reliability_data=exercise_upper_gold, level_of_measurement='interval')
        exercise_upper_gold.pop()
        return krip

    def assert_good_gold(self, staff="upper"):
        score_count = self._d_corpus.score_count()
        gold_score_count = len(self._gold[staff])
        if score_count != gold_score_count:
            raise Exception("Gold count ({0}) does not match score count ({1}).".format(
                gold_score_count, score_count))

    @staticmethod
    def suggestion_data_by_cost(suggestions, costs, details, cutoff_rank=None):
        """
        Return a dictionary of suggestion data keyed by total costs.
        :param suggestions: A list of abcDF suggestion strings, ordered by increasing cost.
        :param costs: A corresponding list of the particular suggestion costs.
        :param details: A corresponding list of cost details for each suggestion
        :param cutoff_rank: The final rank (as indicated by sorting by cost) to include in
        returned results.
        :return: data_by_cost: A dictionary of lists of dictionaries.
        """
        data_by_cost = dict()
        sugg_count = 0
        last_cost_to_include = None
        for i in range(len(suggestions)):
            sugg_count += 1
            sugg = suggestions[i]
            cost = costs[i]
            detail = details[i]

            if last_cost_to_include is not None and cost > last_cost_to_include:
                break

            if cutoff_rank and cutoff_rank >= sugg_count:
                last_cost_to_include = cost

            if cost not in data_by_cost:
                data_by_cost[cost] = list()
            data = {'suggestion': sugg, 'details': detail}
            data_by_cost[cost].append(data)
        return data_by_cost

    def score_advice(self, score_index, staff="upper", cycle=None, last_digit=None, k=10):
        # FIXME: I don't see what this method buys us. generate_advice() should do the right
        # default thing, no?
        if cycle:
            suggestions, costs, details = \
                self._dactyler.generate_advice(staff=staff, score_index=score_index, cycle=cycle, k=k)
        elif last_digit:
            suggestions, costs, details = \
                self._dactyler.generate_advice(staff=staff, score_index=score_index, last_digit=last_digit, k=k)
        else:
            suggestions, costs, details = \
                self._dactyler.generate_advice(staff=staff, score_index=score_index, k=k)
        return suggestions, costs, details

    def _recall_them_all(self, score_index, staff="upper", cycle=None, last_digit=None):
        k = 25
        last_count = None
        while True:
            suggestions, costs, details = self.score_advice(score_index=score_index, staff=staff,
                                                            last_digit=last_digit, cycle=cycle, k=k)
            suggestion_count = len(suggestions)
            if suggestion_count == last_count:
                break
            last_count = suggestion_count
            if self._all_gold_found(score_index=score_index, staff=staff,
                                    last_digit=last_digit, suggestions=suggestions):
                break
            k *= 2
        suggestion_count = len(suggestions)
        if suggestion_count <= 0:
            raise Exception("Bad suggestion count")
        suggestions = self._trim_suggestions(suggestions=suggestions, score_index=score_index,
                                             last_digit=last_digit, staff=staff)
        costs = costs[0:len(suggestions)]
        details = details[0:len(suggestions)]
        return suggestions, costs, details

    @staticmethod
    def estimated_prob_user_happy(suggestion, score_gold):
        prob = 1
        return prob



