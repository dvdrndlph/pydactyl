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
"""
The "Parncutt" class implements and enhances the model described in the
following paper:

   ﻿R. Parncutt, J. A. Sloboda, E. F. Clarke, M. Raekallio, and P. Desain,
       “An ergonomic model of keyboard fingering for melodic fragments,”
       Music Percept., vol. 14, no. 4, pp. 341–382, 1997.     

We enhance the method to handle repeated pitches, two staffs,
and segregated two-hand fingering. Herein, we also implement the "Jacobs"
class, providing the same treatment for the model described here:

   ﻿J. P. Jacobs, “Refinements to the ergonomic model for keyboard
       fingering of Parncutt, Sloboda, Clarke, Raekallio, and Desain,”
       Music Percept., vol. 18, no. 4, pp. 505–511, 2001.

Also included is our own "Badgerow" class, tweaking the Parncutt model
per the suggestions of pianist Justin Badgerow at Elizabethtown College.
"""


import networkx as nx
from itertools import islice
import copy
import re
from . import Dactyler as D
from . import Constant as C
from pydactyl.dcorpus.DNote import DNote
from pydactyl.dcorpus.DAnnotation import DAnnotation


FINGER_SPANS = {
    ('>1', '>1'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>2', '>2'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>3', '>3'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>4', '>4'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>5', '>5'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},

    ('<1', '<1'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<2', '<2'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<3', '<3'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<4', '<4'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<5', '<5'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},

    ('>1', '>2'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    ('>1', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 3, 'MaxRel': 7, 'MaxComf': 10, 'MaxPrac': 12},
    ('>1', '>4'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 5, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    ('>1', '>5'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 13, 'MaxPrac': 15},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},

    ('>2', '>1'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('>3', '>1'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -7, 'MaxRel': -3, 'MaxComf': 2, 'MaxPrac': 4},
    ('>4', '>1'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -5, 'MaxComf': 1, 'MaxPrac': 3},
    ('>5', '>1'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('>3', '>2'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>4', '>2'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('>4', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>3'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>4'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},

    ('<2', '<1'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    ('<3', '<1'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 3, 'MaxRel': 7, 'MaxComf': 10, 'MaxPrac': 12},
    ('<4', '<1'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 5, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    ('<5', '<1'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 13, 'MaxPrac': 15},
    ('<3', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('<4', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('<5', '<2'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('<4', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('<5', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('<5', '<4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},

    ('<1', '<2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('<1', '<3'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -7, 'MaxRel': -3, 'MaxComf': 2, 'MaxPrac': 4},
    ('<1', '<4'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -5, 'MaxComf': 1, 'MaxPrac': 3},
    ('<1', '<5'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('<2', '<3'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<4'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<5'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('<3', '<4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<3', '<5'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<4', '<5'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
}

BADGEROW_FINGER_SPANS = {
    ('>1', '>1'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>2', '>2'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>3', '>3'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>4', '>4'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('>5', '>5'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},

    ('<1', '<1'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<2', '<2'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<3', '<3'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<4', '<4'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},
    ('<5', '<5'): {'MinPrac': 0, 'MinComf': 0, 'MinRel': 0, 'MaxRel': 0, 'MaxComf': 0, 'MaxPrac': 0},

    ('>1', '>2'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 9, 'MaxPrac': 10},
    ('>1', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 1, 'MaxRel': 7, 'MaxComf': 11, 'MaxPrac': 12},
    ('>1', '>4'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 3, 'MaxRel': 9, 'MaxComf': 13, 'MaxPrac': 14},
    ('>1', '>5'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 14, 'MaxPrac': 15},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},

    ('>2', '>1'): {'MinPrac': -10, 'MinComf': -9, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('>3', '>1'): {'MinPrac': -12, 'MinComf': -11, 'MinRel': -7, 'MaxRel': -1, 'MaxComf': 2, 'MaxPrac': 4},
    ('>4', '>1'): {'MinPrac': -14, 'MinComf': -13, 'MinRel': -9, 'MaxRel': -3, 'MaxComf': 1, 'MaxPrac': 3},
    ('>5', '>1'): {'MinPrac': -15, 'MinComf': -14, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('>3', '>2'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>4', '>2'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('>4', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>3'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>4'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},

    ('<2', '<1'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 9, 'MaxPrac': 10},
    ('<3', '<1'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 1, 'MaxRel': 7, 'MaxComf': 11, 'MaxPrac': 12},
    ('<4', '<1'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 3, 'MaxRel': 9, 'MaxComf': 13, 'MaxPrac': 14},
    ('<5', '<1'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 14, 'MaxPrac': 15},
    ('<3', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},
    ('<4', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('<5', '<2'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('<4', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('<5', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('<5', '<4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},

    ('<1', '<2'): {'MinPrac': -10, 'MinComf': -9, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('<1', '<3'): {'MinPrac': -12, 'MinComf': -11, 'MinRel': -7, 'MaxRel': -1, 'MaxComf': 2, 'MaxPrac': 4},
    ('<1', '<4'): {'MinPrac': -14, 'MinComf': -13, 'MinRel': -9, 'MaxRel': -3, 'MaxComf': 1, 'MaxPrac': 3},
    ('<1', '<5'): {'MinPrac': -15, 'MinComf': -14, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('<2', '<3'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<4'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<5'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('<3', '<4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<3', '<5'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<4', '<5'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
}

NOTE_CLASS_IS_BLACK = {
    0: False,
    1: True,
    2: False,
    3: True,
    4: False,
    5: False,
    6: True,
    7: False,
    8: True,
    9: False,
    10: True,
    11: False
}


def is_black(midi_number):
    modulo_number = midi_number % 12
    return NOTE_CLASS_IS_BLACK[modulo_number]


def is_white(midi_number):
    return not is_black(midi_number=midi_number)


def is_between(midi, midi_left, midi_right):
    if not midi or not midi_left or not midi_right:
        return False

    if midi_left < midi < midi_right:
        return True
    if midi_right < midi < midi_left:
        return True

    return False


class Parncutt(D.Dactyler):
    def init_rule_weights(self):
        self._weights = {
            'str': 1,
            'sma': 1,
            'lar': 1,
            'pcc': 1,
            'pcs': 1,
            'wea': 1,
            '345': 1,
            '3t4': 1,
            'bl4': 1,
            'bl1': 1,
            'bl5': 1,
            'pa1': 1
        }

    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive",
                 pruning_method='max', finger_spans=FINGER_SPANS, version=(1,0,0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner,
                         staff_combiner=staff_combiner, version=version)
        # self._finger_spans = FINGER_SPANS
        # if finger_spans:
        self._finger_spans = finger_spans
        self._costs = {}
        self._last_segment_all_paths = None  # Generator of all paths for last segment processed.
        self._pruning_method = None
        self.pruning_method(method=pruning_method)
        self._weights = {}
        self.init_rule_weights()

    def finger_spans(self, finger_spans=None):
        if finger_spans is not None:
            self._finger_spans = finger_spans
        return self._finger_spans

    def last_segment_all_paths(self, all_paths=None):
        if all_paths:
            self._last_segment_all_paths = all_paths
        return self._last_segment_all_paths

    def last_segment_pruned_count(self):
        if self.last_segment_all_paths():
            return len(list(self._last_segment_all_paths))
        return 0

    def pruning_method(self, method=None):
        if method is not None:
            if method not in ('max', 'none', 'min', 'both'):
                raise Exception("Bad pruning method: {0}".format(method))
            self._pruning_method = method
        return self._pruning_method

    def transition_allowed(self, from_midi, from_digit, to_midi, to_digit):
        required_span = to_midi - from_midi

        # Repeated notes are always played with the same finger.
        if required_span == 0:
            if from_digit == to_digit:
                # print("Good {0} to {1} trans of span {2}".format(from_digit, to_digit, required_span))
                return True
            else:
                # print("BAD {0} to {1} trans of span {2}".format(from_digit, to_digit, required_span))
                return False

        # A repeated finger may only be used to play a repeated note in finger legato.
        if from_digit == to_digit:
            if required_span == 0:
                return True
            else:
                return False

        if self.pruning_method() == 'none':
            return True

        if (from_digit, to_digit) not in self._finger_spans:
            # print("BAD {0} to {1} trans of span {2}".format(from_digit, to_digit, required_span))
            return False

        max_prac = self._finger_spans[(from_digit, to_digit)]['MaxPrac']
        min_prac = self._finger_spans[(from_digit, to_digit)]['MinPrac']

        if self.pruning_method() == 'max' and required_span <= max_prac:
            return True
        elif self.pruning_method() == 'min' and required_span >= min_prac:
            return True
        elif min_prac <= required_span <= max_prac:
            return True

        # print("BAD {0} to {1} trans of span {2} (between {3} and {4})".format(from_digit,
                                                                            # to_digit,
                                                                            # required_span,
                                                                            # min_prac,
                                                                            # max_prac))
        return False

    @staticmethod
    def prune_dead_end(g, node_id):
        if node_id == 0:
            return
        if len(list(g.successors(node_id))) > 0:
            return
        predecessor_node_ids = g.predecessors(node_id)
        g.remove_node(node_id)
        for predecessor_id in predecessor_node_ids:
            Parncutt.prune_dead_end(g=g, node_id=predecessor_id)

    def fingered_note_nx_graph(self, segment, hand, handed_first_digit, handed_last_digit):
        g = nx.DiGraph()
        g.add_node(0, start=1, midi=0, digit="-")
        prior_slice_node_ids = list()
        prior_slice_node_ids.append(0)
        last_note_in_segment_index = len(segment) - 1
        note_in_segment_index = 0
        node_id = 1
        on_last_prefingered_note = False
        for note in segment:
            on_first_prefingered_note = False
            slice_node_ids = list()

            if note_in_segment_index == 0 and handed_first_digit:
                on_first_prefingered_note = True

            if note_in_segment_index == last_note_in_segment_index and handed_last_digit:
                on_last_prefingered_note = True

            viable_prior_node_ids = dict()
            for digit in (C.THUMB, C.INDEX, C.MIDDLE, C.RING, C.LITTLE):
                handed_digit = hand + str(digit)
                if on_last_prefingered_note and handed_digit != handed_last_digit:
                    continue
                if on_first_prefingered_note and handed_digit != handed_first_digit:
                    continue
                g.add_node(node_id, midi=note.pitch.midi, digit=handed_digit)
                slice_node_ids.append(node_id)
                if 0 in prior_slice_node_ids:
                    g.add_edge(0, node_id)
                else:
                    incoming_count = 0
                    for prior_node_id in prior_slice_node_ids:
                        prior_node = g.nodes[prior_node_id]
                        prior_midi = prior_node["midi"]
                        prior_handed_digit = prior_node["digit"]
                        if self.transition_allowed(from_midi=prior_midi, from_digit=prior_handed_digit,
                                                   to_midi=note.pitch.midi, to_digit=handed_digit):
                            g.add_edge(prior_node_id, node_id)
                            incoming_count += 1
                            viable_prior_node_ids[prior_node_id] = True
                    if incoming_count == 0:
                        g.remove_node(node_id)
                        slice_node_ids.remove(node_id)
                node_id += 1

            for pni in prior_slice_node_ids:
                if pni not in viable_prior_node_ids:
                    Parncutt.prune_dead_end(g, pni)

            if len(slice_node_ids) > 0:
                prior_slice_node_ids = copy.copy(slice_node_ids)
            else:
                raise Exception("No solution between {0} and {1}".format(
                    handed_first_digit, handed_last_digit))
            note_in_segment_index += 1

        g.add_node(node_id, end=1, midi=0, digit="-")
        for prior_node_id in prior_slice_node_ids:
            g.add_edge(prior_node_id, node_id)

        return g

    def init_costs(self):
        costs = {
            'str': 0,
            'sma': 0,
            'lar': 0,
            'pcc': 0,
            'pcs': 0,
            'wea': 0,
            '345': 0,
            '3t4': 0,
            'bl4': 0,
            'bl1': 0,
            'bl5': 0,
            'pa1': 0,
        }
        return costs

    def distance(self, from_midi, to_midi):
        """
        Estimate the distance between two piano keys identified by MIDI code.
        The original Parncutt paper simply uses semitone differences.
        :param from_midi: The starting piano key.
        :param to_midi: The ending piano key.
        :return: The distance between the two keys.
        """
        return to_midi - from_midi

    @staticmethod
    def _hand_and_trigram_digits(handed_digit_1, handed_digit_2, handed_digit_3):
        pat = re.compile('^([<>])')
        mat = pat.match(handed_digit_2)
        hand = mat.group(1)
        digit_1 = re.sub(pat, '', handed_digit_1)
        digit_2 = re.sub(pat, '', handed_digit_2)
        digit_3 = re.sub(pat, '', handed_digit_3)
        if digit_1 == '-':
            digit_1 = None
        if digit_3 == '-':
            digit_3 = None
        if digit_1 is not None:
            digit_1 = int(digit_1)
        digit_2 = int(digit_2)
        if digit_3 is not None:
            digit_3 = int(digit_3)

        return hand, digit_1, digit_2, digit_3

    def assess_stretch(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        # Rule 1 ("Stretch")
        if not midi_1:
            return

        semitone_diff_12 = self.distance(midi_1, midi_2)
        max_comf_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxComf']
        min_comf_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MinComf']

        # ﻿"Assign 2 points for each semitone that an interval exceeds MaxComf or is less than MinComf."
        if semitone_diff_12 > max_comf_12:
            costs['str'] = 2 * (semitone_diff_12 - max_comf_12) * self._weights['str']
        elif semitone_diff_12 < min_comf_12:
            costs['str'] = 2 * (min_comf_12 - semitone_diff_12) * self._weights['str']

    def assess_small_per_rule(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        # Rule 2 ("Small-Span")
        # "For finger pairs including the thumb, assign 1 point for each semitone that an interval is
        # less than MinRel. For finger pairs not including the thumb, assign 2 points per semitone."
        if not midi_1:
            return

        semitone_diff_12 = self.distance(midi_1, midi_2)
        min_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MinRel']

        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        span_penalty = 2
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = 1
        if semitone_diff_12 < min_rel_12:
            costs['sma'] = span_penalty * (min_rel_12 - semitone_diff_12) * self._weights['sma']

    def assess_small_span(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        # Rule 2 ("Small-Span")
        # "For finger pairs including the thumb, assign 1 point for each semitone that an interval is
        # less than MinRel. For finger pairs not including the thumb, assign 2 points per semitone."
        if not midi_1:
            return

        semitone_diff = self.distance(midi_1, midi_2)
        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        span_penalty = 2
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = 1

        hand = D.Dactyler.digit_hand(handed_digit_1)
        min_rel = None
        max_rel = None
        if hand == '>':
            if digit_1 < digit_2:
                min_rel = self._finger_spans[(handed_digit_1, handed_digit_2)]['MinRel']
            elif digit_1 > digit_2:
                max_rel = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
            else:
                return
        else:
            if digit_1 > digit_2:
                min_rel = self._finger_spans[(handed_digit_1, handed_digit_2)]['MinRel']
            elif digit_1 < digit_2:
                max_rel = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
            else:
                return

        cost = 0
        if min_rel is not None and semitone_diff < min_rel:
            cost = span_penalty * (min_rel - semitone_diff) * self._weights['sma']
        if max_rel is not None and semitone_diff > max_rel:
            cost = span_penalty * (semitone_diff - max_rel) * self._weights['sma']
        costs['sma'] = cost

    def assess_large_span_per_rule(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        # Rule 3 ("Large-Span")
        # "For finger pairs including the thumb, assign 1 point for each semitone that an interval
        # exceeds MaxRel. For finger pairs not including the thumb, assign 2 points per semitone."
        if not midi_1:
            return

        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        span_penalty = 2
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = 1
        semitone_diff_12 = self.distance(midi_1, midi_2)
        max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
        if semitone_diff_12 > max_rel_12:
            costs['lar'] = span_penalty * (semitone_diff_12 - max_rel_12) * self._weights['lar']

    def assess_large_span(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        # Rule 3 ("Large-Span") as described in Parncutt text and implied in results reported,
        # NOT as defined in the stated Rule 3.
        if not midi_1:
            return

        absolute_semitone_diff_12 = abs(self.distance(midi_1, midi_2))
        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        span_penalty = 2
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = 1

        hand = D.Dactyler.digit_hand(handed_digit_1)
        if hand == '>':
            if digit_1 < digit_2 and midi_1 < midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
            elif digit_1 > digit_2 and midi_1 > midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxRel']
            else:
                return
        else:
            if digit_1 < digit_2 and midi_1 < midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxRel']
            elif digit_1 > digit_2 and midi_1 > midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
            else:
                return

        if absolute_semitone_diff_12 > max_rel_12:
            costs['lar'] = span_penalty * (absolute_semitone_diff_12 - max_rel_12) * self._weights['lar']

    def raw_position_change_count(self, handed_digit_1, midi_1, handed_digit_2,
                              midi_2, handed_digit_3, midi_3):
        if not midi_1 or not midi_3:
            return 0

        pcc = 0
        semitone_diff_13 = self.distance(midi_1, midi_3)
        ###if semitone_diff_13 != 0:  # FIXME: This is in the code Parncutt shared and needed to reproduce
        # results for A and E, but is contradicted by Figure 2(iv) example in paper.
        max_comf_13 = self._finger_spans[(handed_digit_1, handed_digit_3)]['MaxComf']
        min_comf_13 = self._finger_spans[(handed_digit_1, handed_digit_3)]['MinComf']
        max_prac_13 = self._finger_spans[(handed_digit_1, handed_digit_3)]['MaxPrac']
        min_prac_13 = self._finger_spans[(handed_digit_1, handed_digit_3)]['MinPrac']

        digit_2 = D.Dactyler.digit_only(handed_digit_2)

        if semitone_diff_13 > max_comf_13:
            if digit_2 == C.THUMB and is_between(midi_2, midi_1, midi_3) and semitone_diff_13 > max_prac_13:
                pcc = 2  # A "full change"
            else:
                pcc = 1  # A "half change"
        elif semitone_diff_13 < min_comf_13:
            if digit_2 == C.THUMB and is_between(midi_2, midi_1, midi_3) and semitone_diff_13 < min_prac_13:
                pcc = 2  # A "full change"
            else:
                pcc = 1 # A "half change"
        return pcc

    def assess_position_change_count(self, costs, handed_digit_1, midi_1, handed_digit_2,
                                     midi_2, handed_digit_3, midi_3):
        # Rule 4 ("Position-Change-Count")
        # "Assign 2 points for every full change of hand position and 1 point for every half change.
        # A change of hand position occurs whenever the first and third notes in a consecutive
        # group of three span an interval that is greater than MaxComf or less than MinComf
        # for the corresponding fingers. In a full change, three conditions are satisfied
        # simultaneously: The finger on the second of the three notes is the thumb; the second pitch
        # lies between the first and third pitches; and the interval between the first and third pitches
        # is greater than MaxPrac or less than MinPrac. All other changes are half changes."
        raw_pcc = self.raw_position_change_count(handed_digit_1, midi_1,
                                                 handed_digit_2, midi_2,
                                                 handed_digit_3, midi_3)
        costs['pcc'] = raw_pcc * self._weights['pcc']

    def assess_position_change_size(self, costs, handed_digit_1, midi_1, handed_digit_3, midi_3):
        # Rule 5 ("Position-Change-Size")
        # "If the interval spanned by the first and third notes in a group of three is less than MinComf,
        # assign the difference between the interval and MinComf (expressed in semitones). Conversely,
        # if the interval is greater than MaxComf, assign the difference between the interval and MaxComf."
        ### if semitone_diff_13 != 0:  # This is in the code Parncutt shared, but is contradicted in paper.
        if not midi_1 or not midi_3:
            return

        semitone_diff_13 = self.distance(midi_1, midi_3)
        max_comf_13 = self._finger_spans[(handed_digit_1, handed_digit_3)]['MaxComf']
        min_comf_13 = self._finger_spans[(handed_digit_1, handed_digit_3)]['MinComf']

        if semitone_diff_13 < min_comf_13:
            costs['pcs'] = (min_comf_13 - semitone_diff_13) * self._weights['pcs']
        elif semitone_diff_13 > max_comf_13:
            costs['pcs'] = (semitone_diff_13 - max_comf_13) * self._weights['pcs']

    def assess_weak_finger(self, costs, digit_2):
        # Rule 6 (wea "Weak-Finger")
        # "Assign 1 point every time finger 4 or finger 5 is used."
        if digit_2 == C.RING or digit_2 == C.LITTLE:
            costs['wea'] = self._weights['wea']

    def assess_345(self, costs, digit_1, digit_2, digit_3):
        # Rule 7 ("Three-Four-Five")
        # "Assign 1 point every time fingers 3, 4, and 5 occur consecutively in any order,
        # even when groups overlap."
        finger_hash = {
            digit_1: True,
            digit_2: True,
            digit_3: True
        }
        if C.MIDDLE in finger_hash and C.RING in finger_hash and C.LITTLE in finger_hash:
            costs['345'] = self._weights['345']

    def assess_3_to_4(self, costs, digit_1, digit_2):
        # Rule 8 ("Three-to-Four")
        # "Assign 1 point each time finger 3 is immediately followed by finger 4."
        if digit_1 == C.MIDDLE and digit_2 == C.RING:
            costs['3t4'] = self._weights['3t4']

    def assess_4_on_black(self, costs, digit_1, midi_1, digit_2, midi_2):
        # Rule 9 ("Four-on-Black")
        # "Assign 1 point each time fingers 3 and 4 occur consecutively in any order with 3 on
        # white and 4 on black."
        if (digit_1 == C.RING and is_black(midi_1) and digit_2 == C.MIDDLE and is_white(midi_2)) or \
                (digit_1 == C.MIDDLE and is_white(midi_1) and digit_2 == C.RING and is_black(midi_2)):
            costs['bl4'] = self._weights['bl4']

    def assess_thumb_on_black(self, costs, digit_1, midi_1, digit_2, midi_2, digit_3, midi_3):
        # Rule 10 ("Thumb-on-Black")
        # "Assign 1 point whenever the thumb plays a black key."
        if digit_2 != C.THUMB or is_white(midi_2):
            return

        costs['bl1'] += self._weights['bl1']

        # "If the immediately preceding note is white, assign a further 2 points."
        if digit_1 and digit_2 == C.THUMB and is_black(midi_2) and is_white(midi_1):
            costs['bl1'] += 2 * self._weights['bl1']

        # "If the immediately following note is white, assign a further 2 points."
        if digit_3 and digit_2 == C.THUMB and is_black(midi_2) and is_white(midi_3):
            costs['bl1'] += 2 * self._weights['bl1']

    def assess_5_on_black(self, costs, midi_1, digit_2, midi_2, midi_3):
        # Rule 11 ("Five-on-Black")
        # "If the fifth finger plays a black key and the immediately preceding and following notes
        # are also black, assign 0 points. If the immediately preceding note is white, assign 2 points.
        # If the immediately following key is white, assign 2 further points."
        black_key_cost = 0
        if digit_2 == C.LITTLE and is_black(midi_2):
            if midi_1 and is_black(midi_1) and midi_3 and is_black(midi_3):
                black_key_cost = 0
            else:
                if midi_1 and is_white(midi_1):
                    black_key_cost = 2
                if midi_3 and is_white(midi_3):
                    black_key_cost += 2
            costs['bl5'] += black_key_cost * self._weights['bl5']

    def assess_thumb_passing(self, costs, hand, digit_1, midi_1, digit_2, midi_2):
        # Rule 12 ("Thumb-Passing")
        # "Assign 1 point for each thumb- or finger-pass on the same level (from white to white
        # or black to black). Assign 3 points if the lower note is white, played by a finger
        # other than the thumb, and the upper is black, played by the thumb." Invert logic for
        # the left hand.
        if not digit_1:
            return

        thumb_passing_cost = 0
        if hand == '>':
            if digit_1 == C.THUMB and midi_2 < midi_1:  # Finger crossing over thumb, descending.
                if (is_white(midi_1) and is_white(midi_2)) or (is_black(midi_1) and is_black(midi_2)):
                    thumb_passing_cost = 1
                elif is_black(midi_1):
                    thumb_passing_cost = 3
                costs['pa1'] = thumb_passing_cost * self._weights['pa1']
            if digit_2 == C.THUMB and midi_2 > midi_1:  # Thumb passing under finger, ascending.
                if (is_white(midi_1) and is_white(midi_2)) or (is_black(midi_1) and is_black(midi_2)):
                    thumb_passing_cost = 1
                elif is_black(midi_2):
                    thumb_passing_cost = 3
                costs['pa1'] = thumb_passing_cost * self._weights['pa1']
        else:
            if digit_1 == C.THUMB and midi_2 > midi_1:  # Finger crossing over thumb, ascending.
                if (is_white(midi_1) and is_white(midi_2)) or (is_black(midi_1) and is_black(midi_2)):
                    thumb_passing_cost = 1
                elif is_black(midi_1):
                    thumb_passing_cost = 3
                costs['pa1'] = thumb_passing_cost * self._weights['pa1']
            if digit_2 == C.THUMB and midi_2 < midi_1:  # Thumb passing under finger, descending.
                if (is_white(midi_1) and is_white(midi_2)) or (is_black(midi_1) and is_black(midi_2)):
                    thumb_passing_cost = 1
                elif is_black(midi_2):
                    thumb_passing_cost = 3
                costs['pa1'] = thumb_passing_cost * self._weights['pa1']

    def trigram_node_cost(self, midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3):
        """
        Determine the cost associated with a trigram node configured as input.
        :param midi_1: The MIDI note number of the first note in the trigram. May be None in first layer.
        :param handed_digit_1: Fingering for first note (e.g., ">3").
        :param midi_2: The MIDI note number of the second note in the trigram.
        :param handed_digit_2: Fingering proposed for second note (e.g., "<5").
        :param midi_3: The MIDI note number of the third note.
        :param handed_digit_3: Fingering for third note.
        :return: cost, costs: The total (scalar integer) cost associated with the node, and a dictionary
        detailing the specific subcosts contributing to the total.
        """
        cost = 0
        costs = self.init_costs()

        hand, digit_1, digit_2, digit_3 = Parncutt._hand_and_trigram_digits(handed_digit_1, handed_digit_2, handed_digit_3)

        # Rule 1 ("Stretch")
        self.assess_stretch(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 2 ("Small-Span")
        self.assess_small_span(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 3 ("Large-Span")
        self.assess_large_span(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 4 ("Position-Change-Count")
        self.assess_position_change_count(costs, handed_digit_1, midi_1, handed_digit_2, midi_2, handed_digit_3, midi_3)

        # Rule 5 ("Position-Change-Size")
        self.assess_position_change_size(costs, handed_digit_1, midi_1, handed_digit_3, midi_3)

        # Rule 6 (wea "Weak-Finger")
        self.assess_weak_finger(costs, digit_2)

        # Rule 7 ("Three-Four-Five")
        self.assess_345(costs, digit_1, digit_2, digit_3)

        # Rule 8 ("Three-to-Four")
        self.assess_3_to_4(costs, digit_1, digit_2)

        # Rule 9 ("Four-on-Black")
        self.assess_4_on_black(costs, digit_1, midi_1, digit_2, midi_2)

        # Rule 10 ("Thumb-on-Black")
        self.assess_thumb_on_black(costs, digit_1, midi_1, digit_2, midi_2, digit_3, midi_3)

        # Rule 11 ("Five-on-Black")
        self.assess_5_on_black(costs, midi_1, digit_2, midi_2, midi_3)

        # Rule 12 ("Thumb-Passing")
        self.assess_thumb_passing(costs, hand, digit_1, midi_1, digit_2, midi_2)

        for cost_key in costs:
            cost += costs[cost_key]
        return cost, costs

    def segment_advice_cost(self, abcdf, staff="upper", score_index=0, segment_index=0):
        """
        Calculate cost and cost details for a given fingering sequence.
        :param abcdf: The fingering sequence.
        :param staff: The staff (one of "upper" or "lower") from which the segment was derived.
        :param score_index: Identifies the score to process.
        :param segment_index: Identifies the segment.
        :return: cost, transition_detail: cost is th total cost. detail is a data structure itemizing
        more granular subcosts.
        """
        segments = self.segments(score_index=score_index, staff=staff)
        segment = segments[segment_index]
        annot = DAnnotation(abcdf=abcdf)
        handed_strikers = annot.handed_strike_digits(staff=staff)
        total_cost = 0
        transition_detail = list()

        if len(segment) < 3:
            # return 0, {}
            raise Exception("Segment too short")

        for note_num in range(len(segment)):
            if note_num == 0:
                midi_1 = None
                hd_1 = '-'
                midi_2 = segment[note_num].pitch.midi
                hd_2 = handed_strikers[note_num]
                midi_3 = segment[note_num + 1].pitch.midi
                hd_3 = handed_strikers[note_num + 1]
            elif note_num == len(segment) - 1:
                midi_1 = segment[note_num - 1].pitch.midi
                hd_1 = handed_strikers[note_num - 1]
                midi_2 = segment[note_num].pitch.midi
                hd_2 = handed_strikers[note_num]
                midi_3 = None
                hd_3 = '-'
            else:
                midi_1 = segment[note_num - 1].pitch.midi
                hd_1 = handed_strikers[note_num - 1]
                midi_2 = segment[note_num].pitch.midi
                hd_2 = handed_strikers[note_num]
                midi_3 = segment[note_num + 1].pitch.midi
                hd_3 = handed_strikers[note_num + 1]

            cost, detail = self.trigram_node_cost(midi_1, hd_1, midi_2, hd_2, midi_3, hd_3)
            total_cost += cost
            transition_detail.append(detail)
        return total_cost, transition_detail

    def trigram_nx_graph(self, fn_graph):
        """
        Generate a trigram trellis graph representation of the fingering problem at hand.
        :param fn_graph: A networkx graph representing the "fingering network" (as in Figure 5 in
        Parncutt paper). Each node contains a handed "digit" and "midi" note number. Notes are connected
        in a trellis, with a single "start" node, a single "end" node, and a number of layers, one for each
        note in the sequence.
        :return: A trigram graph a la Figure 6 in the Parncutt paper.
        """
        g = nx.DiGraph()
        g.add_node(0, uniq='Start', start=1)
        level_1_slice = [0]
        prior_trigram_slice = [0]
        next_trigram_node_id = 1
        done = False
        slice_number = 0
        while not done:
            slice_number += 1
            slice_trigram_id_for_key = dict()
            next_level_1_slice = list()
            for level_1_node_id in level_1_slice:
                level_2_nodes = list(fn_graph.successors(level_1_node_id))
                for level_2_node_id in level_2_nodes:
                    next_level_1_slice.append(level_2_node_id)
                    level_3_nodes = list(fn_graph.successors(level_2_node_id))
                    for level_3_node_id in level_3_nodes:
                        node_1 = fn_graph.nodes[level_1_node_id]
                        node_2 = fn_graph.nodes[level_2_node_id]
                        node_3 = fn_graph.nodes[level_3_node_id]
                        if node_3['digit'] == '-':
                            done = True
                        digit_1 = node_1['digit']
                        digit_2 = node_2['digit']
                        digit_3 = node_3['digit']
                        midi_1 = node_1['midi']
                        midi_2 = node_2['midi']
                        midi_3 = node_3['midi']
                        colored_1 = str(midi_1) + 'b' if is_black(midi_1) else midi_1
                        colored_2 = str(midi_2) + 'b' if is_black(midi_2) else midi_2
                        colored_3 = str(midi_3) + 'b' if is_black(midi_3) else midi_3
                        agg_attr = "{0}: {1}{2}{3}\n{4}/{5}/{6}".format(slice_number,
                                                                        re.sub(r'[<>]', '', digit_1),
                                                                        re.sub(r'[<>]', '', digit_2),
                                                                        re.sub(r'[<>]', '', digit_3),
                                                                        colored_1, colored_2, colored_3)
                        slice_trigram_key = (digit_1, digit_2, digit_3)
                        if slice_trigram_key not in slice_trigram_id_for_key:
                            g.add_node(next_trigram_node_id, uniq=agg_attr,
                                       midi_1=midi_1, digit_1=digit_1,
                                       midi_2=midi_2, digit_2=digit_2,
                                       midi_3=midi_3, digit_3=digit_3)
                            slice_trigram_id_for_key[slice_trigram_key] = next_trigram_node_id
                            trigram_node_id = next_trigram_node_id
                            next_trigram_node_id += 1
                        else:
                            trigram_node_id = slice_trigram_id_for_key[slice_trigram_key]
                        for prior_trigram_node_id in prior_trigram_slice:
                            if 'start' in g.nodes[prior_trigram_node_id] or \
                                (g.nodes[prior_trigram_node_id]['digit_2'] == digit_1 and
                                 g.nodes[prior_trigram_node_id]['digit_3'] == digit_2):
                                weight, weights = self.trigram_node_cost(midi_1=midi_1, handed_digit_1=digit_1,
                                                                         midi_2=midi_2, handed_digit_2=digit_2,
                                                                         midi_3=midi_3, handed_digit_3=digit_3)
                                g.add_edge(prior_trigram_node_id, trigram_node_id, weight=weight, weights=weights)
            level_1_slice = list(set(next_level_1_slice))  # Distinct IDs only
            prior_trigram_slice = []
            for node_key, node_id in slice_trigram_id_for_key.items():
                prior_trigram_slice.append(node_id)

        g.add_node(next_trigram_node_id, q='End')
        for prior_trigram_node_id in prior_trigram_slice:
            g.add_edge(prior_trigram_node_id, next_trigram_node_id, weight=0)

        return g, next_trigram_node_id

    def k_best_advice(self, g, target_id, k):
        """
        Apply standard shortest path algorithms to determine set of optimal fingerings based on
        a standardized networkx graph.
        :param g: The weighted trinode graph. Weights must be specified via a "weight" edge parameter. Fingerings
        must be set on each "handed_digit" node parameter.
        :param target_id: The node id (key) for the last node or end point in the graph.
        :param k: The number of suggestions to return.
        :return: suggestions, costs, details: Three lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second list contains the respective costs of each suggestion. The third
        is a hash table detailing the specific contributions of the various rules to the total suggestion cost.
        """
        if k is None or k == 1:
            rule_costs = dict()
            path = nx.shortest_path(g, source=0, target=target_id, weight="weight")
            segment_abcdf = ''
            for node_id in path:
                node = g.nodes[node_id]
                if "digit_2" in node:
                    segment_abcdf += node["digit_2"]
            cost = nx.shortest_path_length(g, source=0, target=target_id, weight="weight")
            # print("TOTAL COST: {0}".format(cost))
            sub_g = g.subgraph(path)
            for (u, v, weights) in sub_g.edges.data('weights'):
                if not weights:
                    continue
                # print("{0} cost for edge ({1}, {2})".format(weights, u, v))
                for rule_id, rule_cost in weights.items():
                    if rule_id not in rule_costs:
                        rule_costs[rule_id] = 0
                    rule_costs[rule_id] += rule_cost
            return [segment_abcdf], [cost], [rule_costs]
        else:
            sugg_map = dict()
            suggestions = list()
            costs = list()
            details = list()
            k_best_paths = list(islice(nx.shortest_simple_paths(g, source=0, target=target_id, weight="weight"), k))
            for path in k_best_paths:
                rule_costs = dict()
                sub_g = g.subgraph(path)
                suggestion_cost = sub_g.size(weight="weight")
                for (u, v, weights) in sub_g.edges.data('weights'):
                    if not weights:
                        continue
                    for rule_id, rule_cost in weights.items():
                        if rule_id not in rule_costs:
                            rule_costs[rule_id] = 0
                        rule_costs[rule_id] += rule_cost
                segment_abcdf = ''
                for node_id in path:
                    node = g.nodes[node_id]
                    if "digit_2" in node:
                        segment_abcdf += node["digit_2"]
                suggestions.append(segment_abcdf)
                if segment_abcdf in sugg_map:
                    sugg_map[segment_abcdf] += 1
                else:
                    sugg_map[segment_abcdf] = 1
                costs.append(suggestion_cost)
                details.append(rule_costs)

            # print("TOTAL: {0} DISTINCT: {1} COSTS: {2}".format(len(suggestions), len(sugg_map), costs))
            return suggestions, costs, details

    def generate_segment_advice(self, segment, staff, offset=0, cycle=None,
                                handed_first_digit=None, handed_last_digit=None, k=None):
        """
        Generate a set of k ranked fingering suggestions for the given segment.
        :param segment: The segment to work with, as a music21 score object.
        :param staff: The staff (one of "upper" or "lower") from which the segment was derived.
        :param offset: The zero-based index to begin the returned advice.
        :param cycle: Detect repeating note patterns of at least this length within each segment and generate
        advice best suited for uniform fingerings of the repeated patterns. Defaults to None (ignore cycles).
        :param handed_first_digit: Constrain the solution to begin with this finger.
        :param handed_last_digit: Constrain the solution to end with this finger.
        :param k: The number of advice segments to return. The actual number returned may be less,
        but will be no more, than this number.
        :return: suggestions, costs, details: Three lists are returned. The first contains suggested fingering
        solutions as abcDF strings. The second contains the respective costs of each suggestion. The third
        contains details about how each cost was determined.
        """
        if len(segment) == 1:
            note_list = DNote.note_list(segment)
            abcdf = D.Dactyler.one_note_advise(note_list[0], staff=staff,
                                               first_digit=handed_first_digit,
                                               last_digit=handed_last_digit)
            return [abcdf], [0], [0]

        hand = ">"
        if staff == "lower":
            hand = "<"

        k_to_use = k
        # FIXME: This is a hack to get a solutions for
        if cycle:
            first_note = copy.copy(segment[0])
            segment.append(first_note)
            k_to_use = 5 * k

        fn_graph = self.fingered_note_nx_graph(segment=segment, hand=hand,
                                               handed_first_digit=handed_first_digit,
                                               handed_last_digit=handed_last_digit)
        # nx.write_graphml(fn_graph, "/Users/dave/goo.graphml")

        trigram_graph, target_node_id = self.trigram_nx_graph(fn_graph=fn_graph)
        all_paths = nx.all_simple_paths(trigram_graph, source=0, target=target_node_id)
        self.last_segment_all_paths(all_paths)
        # trigram_graphmlized = Parncutt.graphmlize(trigram_graph)
        # nx.write_graphml(trigram_graphmlized, "/Users/dave/gootri.graphml")
        suggestions, costs, details = self.k_best_advice(g=trigram_graph, target_id=target_node_id, k=k_to_use)

        # FIXME too
        if cycle:
            done = False
            good_suggestions = list()
            good_costs = list()
            good_details = list()
            good_count = 0
            last_suggestion_count = 0
            while not done:
                for i in range(len(suggestions)):
                    first_hf = suggestions[i][:2]
                    last_hf = suggestions[i][-2:]
                    if first_hf == last_hf:
                        good_suggestions.append(suggestions[i][:-2])
                        good_costs.append(costs[i])
                        good_details.append(details[i])
                        good_count += 1
                    if good_count == k:
                        # We ignore ties. This may be a mistake. I am pretty sure this is how the networkx
                        # thing we are doing for the normal path does this. FIXME?
                        break
                if good_count == k or len(suggestions) == last_suggestion_count:
                    # We got what was asked for or we got all there was.
                    segment.pop(-1)  # Put it back the way it was.
                    done = True
                else:
                    last_suggestion_count = len(suggestions)
                    good_suggestions = list()
                    good_costs = list()
                    good_details = list()
                    good_count = 0
                    k_to_use *= 2
                    suggestions, costs, details = self.k_best_advice(g=trigram_graph,
                                                                     target_id=target_node_id,
                                                                     k=k_to_use)
            return good_suggestions, good_costs, good_details
        else:
            return suggestions, costs, details


class Jacobs(Parncutt):
    def init_rule_weights(self):
        self._weights = {
            'str': 1,
            'sma': 1,
            'lar': 1,
            'pcc': 1,
            'pcs': 1,
            'wea': 1,
            '3t4': 1,
            'bl4': 1,
            'bl1': 1,
            'bl5': 1,
            'pa1': 1
        }

    def init_costs(self):
        costs = {
            'str': 0,
            'sma': 0,
            'lar': 0,
            'pcc': 0,
            'pcs': 0,
            'wea': 0,
            '3t4': 0,
            'bl4': 0,
            'bl1': 0,
            'bl5': 0,
            'pa1': 0,
        }
        return costs

    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive",
                 pruning_method='max', finger_spans=FINGER_SPANS, version=(1,0,0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner,
                         staff_combiner=staff_combiner, pruning_method=pruning_method,
                         finger_spans=finger_spans, version=version)

        self._key_positions = D.Dactyler.horizontal_key_positions()
        avg_distances = list()
        for interval_size in range(0, 24):
            distance = 0
            for manifestation_num in range(0, 12):
                start_midi = 21 + manifestation_num
                end_midi = start_midi + interval_size
                distance += (self._key_positions[end_midi] - self._key_positions[start_midi])
            avg_distances.append(distance/12)

        self._bounds_for_semitone_interval = list()
        self._bounds_for_semitone_interval.append(0)

        for i in range(1, len(avg_distances)):
            if i == 1:
                self._bounds_for_semitone_interval.append(0)
            else:
                self._bounds_for_semitone_interval.append((avg_distances[i] + avg_distances[i-1])/2.0)

    def distance(self, from_midi, to_midi):
        from_pos = self._key_positions[from_midi]
        to_pos = self._key_positions[to_midi]
        multiplier = 1
        dist = to_pos - from_pos
        if to_midi < from_midi:
            multiplier = -1
            dist = from_pos - to_pos
        for i in range(len(self._bounds_for_semitone_interval) - 1):
            if self._bounds_for_semitone_interval[i] <= dist <= self._bounds_for_semitone_interval[i+1]:
                return multiplier * i
        raise Exception("Distance between {0} and {1} could not be calculated".format(from_midi, to_midi))

    def assess_weak_finger(self, costs, digit_2):
        # Rule 6 (wea "Weak-Finger")
        # Assign 1 point every time finger 4 is used (but no longer finger 5).
        if digit_2 == C.RING:
            costs['wea'] = self._weights['wea']

    def assess_large_span(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        # Rule 3 ("Large-Span") as described in Parncutt text and implied in results reported,
        # NOT as defined in the stated Rule 3.
        if not midi_1:
            return

        absolute_semitone_diff_12 = abs(self.distance(midi_1, midi_2))
        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        # "penalizes equally (by 1 point per semitone) all spans exceeding MaxRel."
        span_penalty = 1

        hand = D.Dactyler.digit_hand(handed_digit_1)
        if hand == '>':
            if digit_1 < digit_2 and midi_1 < midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
            elif digit_1 > digit_2 and midi_1 > midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxRel']
            else:
                return
        else:
            if digit_1 < digit_2 and midi_1 < midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxRel']
            elif digit_1 > digit_2 and midi_1 > midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
            else:
                return

        if absolute_semitone_diff_12 > max_rel_12:
            costs['lar'] = span_penalty * (absolute_semitone_diff_12 - max_rel_12) * self._weights['lar']

    def trigram_node_cost(self, midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3):
        """
        Determine the cost associated with a trigram node configured as input.
        :param midi_1: The MIDI note number of the first note in the trigram. May be None in first layer.
        :param handed_digit_1: Fingering for first note (e.g., ">3").
        :param midi_2: The MIDI note number of the second note in the trigram.
        :param handed_digit_2: Fingering proposed for second note (e.g., "<5").
        :param midi_3: The MIDI note number of the third note.
        :param handed_digit_3: Fingering for third note.
        :return: cost, costs: The total (scalar integer) cost associated with the node, and a dictionary
        detailing the specific subcosts contributing to the total.
        """
        cost = 0
        costs = self.init_costs()

        hand, digit_1, digit_2, digit_3 = Parncutt._hand_and_trigram_digits(handed_digit_1, handed_digit_2, handed_digit_3)

        # Rule 1 ("Stretch")
        self.assess_stretch(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 2 ("Small-Span")
        self.assess_small_span(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 3 ("Large-Span")
        self.assess_large_span(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 4 ("Position-Change-Count")
        self.assess_position_change_count(costs, handed_digit_1, midi_1, handed_digit_2, midi_2, handed_digit_3, midi_3)

        # Rule 5 ("Position-Change-Size")
        self.assess_position_change_size(costs, handed_digit_1, midi_1, handed_digit_3, midi_3)

        # Rule 6 (wea "Weak-Finger")
        self.assess_weak_finger(costs, digit_2)

        # Rule 7 ("Three-Four-Five")
        # Not done in the interests of parsomony.
        # self.assess_345(costs, digit_1, digit_2, digit_3)

        # Rule 8 ("Three-to-Four")
        self.assess_3_to_4(costs, digit_1, digit_2)

        # Rule 9 ("Four-on-Black")
        self.assess_4_on_black(costs, digit_1, midi_1, digit_2, midi_2)

        # Rule 10 ("Thumb-on-Black")
        self.assess_thumb_on_black(costs, digit_1, midi_1, digit_2, midi_2, digit_3, midi_3)

        # Rule 11 ("Five-on-Black")
        self.assess_5_on_black(costs, midi_1, digit_2, midi_2, midi_3)

        # Rule 12 ("Thumb-Passing")
        self.assess_thumb_passing(costs, hand, digit_1, midi_1, digit_2, midi_2)

        for cost_key in costs:
            cost += costs[cost_key]
        return cost, costs


class Badgerow(Parncutt):
    def init_rule_weights(self):
        self._weights = {
            'str': 1,
            'sma': 1,
            'lar': 1,
            'pcc': 1,
            'pcs': 1,
            'wea': 1,
            '345': 1,
            '3t4': 1,
            'bl1': 1,
            'bl5': 1,
            'pa1': 1,
            'apr': 1,
            'afc': 1,
            # 'wb1': 1,
            'b1w': 1,
        }

    def init_costs(self):
        costs = {
            'str': 0,
            'sma': 0,
            'lar': 0,
            'pcc': 0,
            'pcs': 0,
            'wea': 0,
            '345': 0,
            '3t4': 0,
            'bl1': 0,
            'bl5': 0,
            'pa1': 0,
            'apr': 0,
            'afc': 0,
            # 'wb1': 0,
            'b1w': 0,
        }
        return costs

    def assess_large_span_badgerow(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2, handed_digit_3, midi_3):
        # Rule 3 ("Large-Span") as described in Parncutt text and implied in results reported,
        # NOT as defined in the stated Rule 3. Amended as suggested by Badgerow:
        #
        # "If PCC (Position Change Count) is less than or equal to 1, assign points exceeding
        # MaxComf, not MaxRel. So, for finger pairs including the thumb, assign 1 point for
        # each semitone that an interval exceeds MaxComf. For finger pairs not including
        # the thumb, assign 2 points per semitone than an interval exceeds MaxComf."
        #
        if not midi_1:
            return

        absolute_semitone_diff_12 = abs(self.distance(midi_1, midi_2))
        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        span_penalty = 2
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = 1

        hand = D.Dactyler.digit_hand(handed_digit_1)
        if hand == '>':
            if digit_1 < digit_2 and midi_1 < midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
                max_comf_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxComf']
            elif digit_1 > digit_2 and midi_1 > midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxRel']
                max_comf_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxComf']
            else:
                return
        else:
            if digit_1 < digit_2 and midi_1 < midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxRel']
                max_comf_12 = self._finger_spans[(handed_digit_2, handed_digit_1)]['MaxComf']
            elif digit_1 > digit_2 and midi_1 > midi_2:
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
                max_comf_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxComf']
            else:
                return

        raw_pcc = 2
        if midi_3:
            raw_pcc = self.raw_position_change_count(handed_digit_1, midi_1, handed_digit_2, midi_2,
                                                     handed_digit_3, midi_3)
        if raw_pcc <= 1:
            if absolute_semitone_diff_12 > max_comf_12:
                costs['lar'] = span_penalty * (absolute_semitone_diff_12 - max_comf_12) * self._weights['lar']
        else:
            if absolute_semitone_diff_12 > max_rel_12:
                costs['lar'] = span_penalty * (absolute_semitone_diff_12 - max_rel_12) * self._weights['lar']

    def assess_alternation_pairing(self, costs, digit_1, digit_2, digit_3):
        # New Rule ("Alternation-Pairing") from Justin Badgerow
        # "Assign 1 point for 3-4-3 or 4-3-4 combinations and 1 point for 4-5-4 or 5-4-5 combinations."
        if (digit_1 == 3 and digit_2 == 4 and digit_3 == 3) or (digit_1 == 4 and digit_2 == 3 and digit_3 == 4) or \
           (digit_1 == 4 and digit_2 == 5 and digit_3 == 4) or (digit_1 == 5 and digit_2 == 4 and digit_3 == 5):
            costs['apr'] = self._weights['apr']

    def assess_alternation_finger_change(self, costs, digit_1, midi_1, digit_3, midi_3):
        # New Rule ("Alternation-Finger-Change") from Justin Badgerow
        # "Where the first and third notes are the same, assign 1 point when a different finger
        # is used for these two notes."
        if digit_1 and digit_3 and midi_1 == midi_3 and digit_1 != digit_3:
            costs['afc'] = self._weights['afc']

    def assess_thumb_on_black_to_weak(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        if midi_1 == midi_2:
            return
        digit_1 = D.Dactyler.digit_only(handed_digit_1) if midi_1 else None
        if digit_1 != C.THUMB or is_white(midi_1):
            return
        digit_2 = D.Dactyler.digit_only(handed_digit_2)
        if is_black(midi_2) or digit_2 not in (C.RING, C.LITTLE):
            return
        hand_1 = D.Dactyler.digit_hand(handed_digit_1)
        hand_2 = D.Dactyler.digit_hand(handed_digit_2)
        if hand_1 != hand_2:
            return

        # ANY descending move from 1-4 or 1-5 from black to white key, regardless of size,
        # will be penalized an extra 2 points for 1-4 pairs and an extra 3 points for 1-5 pairs.
        # Assess the same penalties for ascending intervals, if said intervals are less
        # than MinRel. And flip this around for the left hand.

        if hand_1 == '>':
            if midi_1 > midi_2:  # descending
                costs['b1w'] += (digit_2 - 2) * self._weights['b1w']
            else:
                distance = self.distance(midi_1, midi_2)
                min_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MinRel']
                if distance < min_rel_12:
                    costs['b1w'] += (digit_2 - 2) * self._weights['b1w']
        else:  # Left hand
            if midi_2 > midi_1:  # ascending
                costs['b1w'] += (digit_2 - 2) * self._weights['b1w']
            else:
                distance = self.distance(midi_1, midi_2)
                min_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MinRel']
                if distance < min_rel_12:
                    costs['b1w'] += (digit_2 - 2) * self._weights['b1w']

    def assess_weak_to_thumb_on_black(self, costs, handed_digit_1, midi_1, handed_digit_2, midi_2):
        if not midi_1 or midi_1 == midi_2:
            return
        digit_2 = D.Dactyler.digit_only(handed_digit_2) if handed_digit_2 else None
        if digit_2 != C.THUMB or is_white(midi_2):
            return
        digit_1 = D.Dactyler.digit_only(handed_digit_1)
        if is_black(midi_1) or digit_1 not in (C.RING, C.LITTLE):
            return
        hand_1 = D.Dactyler.digit_hand(handed_digit_1)
        hand_2 = D.Dactyler.digit_hand(handed_digit_2)
        if hand_1 != hand_2:
            return

        # For any descending move from 4-1 or 5-1, from black to white key, regardless of size,
        # assess 2 points for 4-1 pairs and 3 points for 5-1 pairs.
        # Assess the same penalties for descending intervals, if said intervals are more
        # than MaxRel.
        if hand_1 == '>':
            if midi_2 > midi_1:  # ascending
                costs['wb1'] += (digit_1 - 2) * self._weights['wb1']
            else:
                distance = self.distance(midi_1, midi_2)
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
                if distance > max_rel_12:
                    costs['wb1'] += (digit_1 - 2) * self._weights['wb1']
        else:  # Left hand
            if midi_2 > midi_1:  # ascending
                distance = self.distance(midi_1, midi_2)
                max_rel_12 = self._finger_spans[(handed_digit_1, handed_digit_2)]['MaxRel']
                if distance > max_rel_12:
                    costs['wb1'] += (digit_1 - 2) * self._weights['wb1']
            else:
                costs['wb1'] += (digit_1 - 2) * self._weights['wb1']

    # def assess_thumb_on_black(self, costs, digit_1, midi_1, digit_2, midi_2, digit_3, midi_3):
    #     # Rule 10 ("Thumb-on-Black")
    #     # "Assign 1 point whenever the thumb plays a black key."
    #     if digit_2 != C.THUMB or is_white(midi_2):
    #         return
    #
    #     costs['bl1'] += self._weights['bl1']
    #
    #     # "If the immediately preceding note is white, assign a further 2 points."
    #     if digit_1 and digit_2 == C.THUMB and is_black(midi_2) and is_white(midi_1):
    #         costs['bl1'] += 2 * self._weights['bl1']
    #
    #     # "If the immediately following note is white, assign a further 2 points."
    #     if digit_3 and digit_2 == C.THUMB and is_black(midi_2) and is_white(midi_3):
    #         costs['bl1'] += 2 * self._weights['bl1']
    #
    #         # Justin's amendment: "When the thumb plays a black key, if the preceding OR
    #         # following note is finger 5 on a white key, assign a further 2 points for each usage."
    #         if digit_1 == C.LITTLE and is_white(midi_1):
    #             costs['bl1'] += 2 * self._weights['bl1']
    #         if digit_3 == C.LITTLE and is_white(midi_3):
    #             costs['bl1'] += 2 * self._weights['bl1']

    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive",
                 pruning_method='max', finger_spans=BADGEROW_FINGER_SPANS, version=(1,0,0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner,
                         staff_combiner=staff_combiner, pruning_method=pruning_method,
                         finger_spans=finger_spans, version=version)

    def trigram_node_cost(self, midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3):
        """
        Determine the cost associated with a trigram node configured as input.
        :param midi_1: The MIDI note number of the first note in the trigram. May be None in first layer.
        :param handed_digit_1: Fingering for first note (e.g., ">3").
        :param midi_2: The MIDI note number of the second note in the trigram.
        :param handed_digit_2: Fingering proposed for second note (e.g., "<5").
        :param midi_3: The MIDI note number of the third note.
        :param handed_digit_3: Fingering for third note.
        :return: cost, costs: The total (scalar integer) cost associated with the node, and a dictionary
        detailing the specific subcosts contributing to the total.
        """
        cost = 0
        costs = self.init_costs()

        hand, digit_1, digit_2, digit_3 = Parncutt._hand_and_trigram_digits(handed_digit_1, handed_digit_2, handed_digit_3)

        # Rule 1 ("Stretch")
        self.assess_stretch(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 2 ("Small-Span")
        self.assess_small_span(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        # Rule 3 ("Large-Span")
        self.assess_large_span_badgerow(costs, handed_digit_1, midi_1, handed_digit_2, midi_2, handed_digit_3, midi_3)

        # Rule 4 ("Position-Change-Count")
        self.assess_position_change_count(costs, handed_digit_1, midi_1, handed_digit_2, midi_2, handed_digit_3, midi_3)

        # Rule 5 ("Position-Change-Size")
        self.assess_position_change_size(costs, handed_digit_1, midi_1, handed_digit_3, midi_3)

        # Rule 6 (wea "Weak-Finger")
        self.assess_weak_finger(costs, digit_2)

        # Rule 7 ("Three-Four-Five")
        self.assess_345(costs, digit_1, digit_2, digit_3)

        # Rule 8 ("Three-to-Four")
        self.assess_3_to_4(costs, digit_1, digit_2)

        # Rule 10 ("Thumb-on-Black")
        self.assess_thumb_on_black(costs, digit_1, midi_1, digit_2, midi_2, digit_3, midi_3)

        # Rule 11 ("Five-on-Black")
        self.assess_5_on_black(costs, midi_1, digit_2, midi_2, midi_3)

        # Rule 12 ("Thumb-Passing")
        self.assess_thumb_passing(costs, hand, digit_1, midi_1, digit_2, midi_2)

        # New rule ("Alternation-Pairing")
        self.assess_alternation_pairing(costs, digit_1, digit_2, digit_3)

        # New rule ("Alternation-Finger-Change")
        self.assess_alternation_finger_change(costs, digit_1, midi_1, digit_3, midi_3)

        # New rule ("Thumb-on-Black-to-Weak")
        self.assess_thumb_on_black_to_weak(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)
        # New rule ("Weak-to-Thumb-on-Black")
        # self.assess_weak_to_thumb_on_black(costs, handed_digit_1, midi_1, handed_digit_2, midi_2)

        for cost_key in costs:
            cost += costs[cost_key]
        return cost, costs
