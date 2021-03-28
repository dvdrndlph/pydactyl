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
""" Implements a reproducible random fingering model.
"""

import random
import copy
import os

from pydactyl.dactyler import Constant
from . import Dactyler as D
BIG_NUM = 999
MAX_INTERVAL_SIZE = 12
BIN_DIR = os.path.abspath(os.path.dirname(__file__))
COST_FILE = Constant.DATA_DIR + '/tables_0.dat'


class Random(D.Dactyler):
    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive",
                 cost_path=None, max_interval_size=MAX_INTERVAL_SIZE, version=(1, 0, 0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner,
                         staff_combiner=staff_combiner, version=version)
        self._cost_path = COST_FILE
        if cost_path:
            self._cost_path = cost_path
        self._max_interval_size = max_interval_size
        self._costs = []  # self._define_costs()

    @staticmethod
    def segment_advice_cost(abcdf, staff="upper", score_index=0, segment_index=0):
        """
        NOT YET IMPLEMENTED
        Calculate cost and cost details for a given fingering sequence.
        :param abcdf: The fingering sequence.
        :param staff: The staff (one of "upper" or "lower") from which the segment was derived.
        :param score_index: Identifies the score to process.
        :param segment_index: Identifies the segment.
        :return: cost, transition_detail: cost is the total cost. detail is a data structure itemizing
        more granular subcosts.
        """
        raise Exception("Not yet implemented.")

    def generate_segment_advice(self, segment, staff, offset=0, cycle=False,
                                handed_first_digit=None, handed_last_digit=None, k=None):
        """
        Generate a set of k ranked fingering suggestions for the given segment. Note that the original
        Hart implementation only returns one best fingering.
        :param segment: The segment to work with, as a music21 score m21_object.
        :param staff: The staff (one of "upper" or "lower") from which the segment was derived.
        :param offset: The zero-based index to begin the returned advice.
        :param cycle: Treat the segment as a repeating pattern and generate advice best suited to
        being repeated. Defaults to False.
        :param handed_first_digit: Constrain the solution to begin with this finger.
        :param handed_last_digit: Constrain the solution to end with this finger.
        :param k: The number of advice segments to return. The actual number returned may be less,
        but will be no more, than this number.
        :return: suggestions, costs, details: Three lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second contains the respective costs of each suggestion. The third
        contains details on how each cost was determined.
        """
        hand = ">"
        if staff == "lower":
            hand = "<"

        k_to_use = k
        # FIXME: This is a hack to get a solutions for
        if cycle:
            first_note = copy.copy(segment[0])
            segment.append(first_note)
            k_to_use = 5 * k

        random.seed(2020171977876576)
        note_count = len(segment)
        suggestions = []
        costs = []
        details = []
        cost = 1
        for s in range(k_to_use):
            abcdf = hand
            for n in range(note_count):
                abcdf += str(random.randint(1, 5))
            suggestions.append(abcdf)
            costs.append(cost)
            details.append(cost)
            cost += 1

        return suggestions, costs, details
