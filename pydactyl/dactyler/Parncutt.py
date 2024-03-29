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

   R. Parncutt, J. A. Sloboda, E. F. Clarke, M. Raekallio, and P. Desain,
       “An ergonomic model of keyboard fingering for melodic fragments,”
       Music Percept., vol. 14, no. 4, pp. 341–382, 1997.     

We enhance the method to handle repeated pitches, two staffs,
and segregated two-hand fingering. Herein, we also implement the "Jacobs"
class, providing the same treatment for the model described here:

   J. P. Jacobs, “Refinements to the ergonomic model for keyboard
       fingering of Parncutt, Sloboda, Clarke, Raekallio, and Desain,”
       Music Percept., vol. 18, no. 4, pp. 505–511, 2001.

We also approximate the methods from

   Balliauw, M., Herremans, D., Palhazi Cuervo, D., & Sörensen, K. (2017).
       A variable neighborhood search algorithm to generate piano fingerings for 
       polyphonic sheet music. International Transactions in Operational Research, 
       24(3), 509–535. https://doi.org/10.1111/itor.12211
       
Also included is our own "Badgerow" class, tweaking the Parncutt model
per the suggestions of pianist Justin Badgerow at Elizabethtown College.
"""

from abc import ABC, abstractmethod
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

BALLIAUW_LARGE_FINGER_SPANS = {
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

    ('>1', '>2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': 1, 'MaxRel': 6, 'MaxComf': 9, 'MaxPrac': 11},
    ('>1', '>3'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': 3, 'MaxRel': 9, 'MaxComf': 13, 'MaxPrac': 15},
    ('>1', '>4'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': 5, 'MaxRel': 11, 'MaxComf': 14, 'MaxPrac': 16},
    ('>1', '>5'): {'MinPrac': -2, 'MinComf': 0, 'MinRel': 7, 'MaxRel': 12, 'MaxComf': 16, 'MaxPrac': 18},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 5, 'MaxPrac': 7},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 10, 'MaxPrac': 12},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},

    ('>2', '>1'): {'MinPrac': -11, 'MinComf': -9, 'MinRel': -6, 'MaxRel': -1, 'MaxComf': 8, 'MaxPrac': 10},
    ('>3', '>1'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -9, 'MaxRel': -3, 'MaxComf': 6, 'MaxPrac': 8},
    ('>4', '>1'): {'MinPrac': -16, 'MinComf': -14, 'MinRel': -11, 'MaxRel': -5, 'MaxComf': 4, 'MaxPrac': 6},
    ('>5', '>1'): {'MinPrac': -18, 'MinComf': -16, 'MinRel': -12, 'MaxRel': -7, 'MaxComf': 0, 'MaxPrac': 2},
    ('>3', '>2'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>4', '>2'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>2'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('>4', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>3'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>4'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},

    ('<2', '<1'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': 1, 'MaxRel': 6, 'MaxComf': 9, 'MaxPrac': 11},
    ('<3', '<1'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': 3, 'MaxRel': 9, 'MaxComf': 13, 'MaxPrac': 15},
    ('<4', '<1'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': 5, 'MaxRel': 11, 'MaxComf': 14, 'MaxPrac': 16},
    ('<5', '<1'): {'MinPrac': -2, 'MinComf': 0, 'MinRel': 7, 'MaxRel': 12, 'MaxComf': 16, 'MaxPrac': 18},
    ('<3', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 5, 'MaxPrac': 7},
    ('<4', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('<5', '<2'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 10, 'MaxPrac': 12},
    ('<4', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('<5', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('<5', '<4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},

    ('<1', '<2'): {'MinPrac': -11, 'MinComf': -9, 'MinRel': -6, 'MaxRel': -1, 'MaxComf': 8, 'MaxPrac': 10},
    ('<1', '<3'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -9, 'MaxRel': -3, 'MaxComf': 6, 'MaxPrac': 8},
    ('<1', '<4'): {'MinPrac': -16, 'MinComf': -14, 'MinRel': -11, 'MaxRel': -5, 'MaxComf': 4, 'MaxPrac': 6},
    ('<1', '<5'): {'MinPrac': -18, 'MinComf': -16, 'MinRel': -12, 'MaxRel': -7, 'MaxComf': 0, 'MaxPrac': 2},
    ('<2', '<3'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<4'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<5'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('<3', '<4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<3', '<5'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<4', '<5'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
}

BALLIAUW_SMALL_FINGER_SPANS = {
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

    ('>1', '>2'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': 1, 'MaxRel': 3, 'MaxComf': 8, 'MaxPrac': 10},
    ('>1', '>3'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': 3, 'MaxRel': 6, 'MaxComf': 10, 'MaxPrac': 12},
    ('>1', '>4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 5, 'MaxRel': 8, 'MaxComf': 11, 'MaxPrac': 13},
    ('>1', '>5'): {'MinPrac': -2, 'MinComf': 0, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 12, 'MaxPrac': 14},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},

    ('>2', '>1'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -3, 'MaxRel': -1, 'MaxComf': 5, 'MaxPrac': 7},
    ('>3', '>1'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -6, 'MaxRel': -3, 'MaxComf': 4, 'MaxPrac': 6},
    ('>4', '>1'): {'MinPrac': -13, 'MinComf': -11, 'MinRel': -8, 'MaxRel': -5, 'MaxComf': 2, 'MaxPrac': 4},
    ('>5', '>1'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': 0, 'MaxPrac': 2},
    ('>3', '>2'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>4', '>2'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('>4', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>3'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>4'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},

    ('<2', '<1'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': 1, 'MaxRel': 3, 'MaxComf': 8, 'MaxPrac': 10},
    ('<3', '<1'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': 3, 'MaxRel': 6, 'MaxComf': 10, 'MaxPrac': 12},
    ('<4', '<1'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': 5, 'MaxRel': 8, 'MaxComf': 11, 'MaxPrac': 13},
    ('<5', '<1'): {'MinPrac': -2, 'MinComf': 0, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 12, 'MaxPrac': 14},
    ('<3', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},
    ('<4', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('<5', '<2'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('<4', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('<5', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('<5', '<4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},

    ('<1', '<2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -3, 'MaxRel': -1, 'MaxComf': 5, 'MaxPrac': 7},
    ('<1', '<3'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -6, 'MaxRel': -3, 'MaxComf': 4, 'MaxPrac': 6},
    ('<1', '<4'): {'MinPrac': -13, 'MinComf': -11, 'MinRel': -8, 'MaxRel': -5, 'MaxComf': 2, 'MaxPrac': 4},
    ('<1', '<5'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': 0, 'MaxPrac': 2},
    ('<2', '<3'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<4'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<5'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('<3', '<4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<3', '<5'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<4', '<5'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
}

BALLIAUW_MEDIUM_FINGER_SPANS = {
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

    ('>1', '>2'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    ('>1', '>3'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': 3, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    ('>1', '>4'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 5, 'MaxRel': 11, 'MaxComf': 13, 'MaxPrac': 15},
    ('>1', '>5'): {'MinPrac': -2, 'MinComf': 0, 'MinRel': 7, 'MaxRel': 12, 'MaxComf': 14, 'MaxPrac': 16},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 5, 'MaxPrac': 7},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 10, 'MaxPrac': 12},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},

    ('>2', '>1'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 6, 'MaxPrac': 8},
    ('>3', '>1'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -3, 'MaxComf': 5, 'MaxPrac': 7},
    ('>4', '>1'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -11, 'MaxRel': -5, 'MaxComf': 3, 'MaxPrac': 5},
    ('>5', '>1'): {'MinPrac': -16, 'MinComf': -14, 'MinRel': -12, 'MaxRel': -7, 'MaxComf': 0, 'MaxPrac': 2},
    ('>3', '>2'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>4', '>2'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>2'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('>4', '>3'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>3'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('>5', '>4'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},

    ('<2', '<1'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': 1, 'MaxRel': 5, 'MaxComf': 8, 'MaxPrac': 10},
    ('<3', '<1'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': 3, 'MaxRel': 9, 'MaxComf': 12, 'MaxPrac': 14},
    ('<4', '<1'): {'MinPrac': -5, 'MinComf': -3, 'MinRel': 5, 'MaxRel': 11, 'MaxComf': 13, 'MaxPrac': 15},
    ('<5', '<1'): {'MinPrac': -2, 'MinComf': 0, 'MinRel': 7, 'MaxRel': 12, 'MaxComf': 14, 'MaxPrac': 16},
    ('<3', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 5, 'MaxPrac': 7},
    ('<4', '<2'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('<5', '<2'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 10, 'MaxPrac': 12},
    ('<4', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('<5', '<3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 6, 'MaxPrac': 8},
    ('<5', '<4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 4, 'MaxPrac': 6},

    ('<1', '<2'): {'MinPrac': -10, 'MinComf': -8, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 6, 'MaxPrac': 8},
    ('<1', '<3'): {'MinPrac': -14, 'MinComf': -12, 'MinRel': -9, 'MaxRel': -3, 'MaxComf': 5, 'MaxPrac': 7},
    ('<1', '<4'): {'MinPrac': -15, 'MinComf': -13, 'MinRel': -11, 'MaxRel': -5, 'MaxComf': 3, 'MaxPrac': 5},
    ('<1', '<5'): {'MinPrac': -16, 'MinComf': -14, 'MinRel': -12, 'MaxRel': -7, 'MaxComf': 0, 'MaxPrac': 2},
    ('<2', '<3'): {'MinPrac': -7, 'MinComf': -5, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<4'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<2', '<5'): {'MinPrac': -12, 'MinComf': -10, 'MinRel': -6, 'MaxRel': -5, 'MaxComf': -2, 'MaxPrac': -2},
    ('<3', '<4'): {'MinPrac': -4, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
    ('<3', '<5'): {'MinPrac': -8, 'MinComf': -6, 'MinRel': -4, 'MaxRel': -3, 'MaxComf': -1, 'MaxPrac': -1},
    ('<4', '<5'): {'MinPrac': -6, 'MinComf': -4, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
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
    ('>1', '>4'): {'MinPrac': -3, 'MinComf': -1, 'MinRel': 2, 'MaxRel': 9, 'MaxComf': 13, 'MaxPrac': 14},
    ('>1', '>5'): {'MinPrac': -1, 'MinComf': 1, 'MinRel': 7, 'MaxRel': 10, 'MaxComf': 14, 'MaxPrac': 15},
    ('>2', '>3'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 5},
    ('>2', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>2', '>5'): {'MinPrac': 2, 'MinComf': 2, 'MinRel': 5, 'MaxRel': 6, 'MaxComf': 8, 'MaxPrac': 10},
    ('>3', '>4'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 2, 'MaxPrac': 4},
    ('>3', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 3, 'MaxRel': 4, 'MaxComf': 5, 'MaxPrac': 7},
    ('>4', '>5'): {'MinPrac': 1, 'MinComf': 1, 'MinRel': 1, 'MaxRel': 2, 'MaxComf': 3, 'MaxPrac': 5},

    ('>2', '>1'): {'MinPrac': -10, 'MinComf': -9, 'MinRel': -5, 'MaxRel': -1, 'MaxComf': 3, 'MaxPrac': 5},
    ('>3', '>1'): {'MinPrac': -12, 'MinComf': -11, 'MinRel': -7, 'MaxRel': -1, 'MaxComf': 2, 'MaxPrac': 4},
    ('>4', '>1'): {'MinPrac': -14, 'MinComf': -13, 'MinRel': -9, 'MaxRel': -2, 'MaxComf': 1, 'MaxPrac': 3},
    ('>5', '>1'): {'MinPrac': -15, 'MinComf': -14, 'MinRel': -10, 'MaxRel': -7, 'MaxComf': -1, 'MaxPrac': 1},
    ('>3', '>2'): {'MinPrac': -5, 'MinComf': -2, 'MinRel': -2, 'MaxRel': -1, 'MaxComf': -1, 'MaxPrac': -1},
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


class TrigramNode(ABC):
    def _digits(self):
        pat = re.compile('^([<>])')
        digit_1 = re.sub(pat, '', self.handed_digit_1)
        digit_2 = re.sub(pat, '', self.handed_digit_2)
        digit_3 = re.sub(pat, '', self.handed_digit_3)
        if digit_1 == '-':
            digit_1 = None
        if digit_3 == '-':
            digit_3 = None
        if digit_1 is not None:
            digit_1 = int(digit_1)
        digit_2 = int(digit_2)
        if digit_3 is not None:
            digit_3 = int(digit_3)
        return digit_1, digit_2, digit_3

    def _hands(self):
        pat = re.compile('^([<>])')

        hand_1 = None
        mat = pat.match(self.handed_digit_1)
        if mat:
            hand_1 = mat.group(1)

        hand_2 = None
        mat = pat.match(self.handed_digit_2)
        if mat:
            hand_2 = mat.group(1)

        hand_3 = None
        mat = pat.match(self.handed_digit_3)
        if mat:
            hand_3 = mat.group(1)

        return hand_1, hand_2, hand_3

    def same_hands(self):
        hands = self._hands()
        if hands[0] == hands[1] == hands[2]:
            return True
        return False

    def __init__(self, midi_1, handed_digit_1, midi_2, handed_digit_2, midi_3, handed_digit_3):
        """
        Initialize a TrigramNode object.
        :param midi_1: The MIDI note number of the first note in the trigram. May be None in first layer.
        :param handed_digit_1: Fingering for first note (e.g., ">3").
        :param midi_2: The MIDI note number of the second note in the trigram.
        :param handed_digit_2: Fingering proposed for second note (e.g., "<5").
        :param midi_3: The MIDI note number of the third note.
        :param handed_digit_3: Fingering for third note.
        """
        self.midi_1 = midi_1
        self.midi_2 = midi_2
        self.midi_3 = midi_3
        self.handed_digit_1 = handed_digit_1
        self.handed_digit_2 = handed_digit_2
        self.handed_digit_3 = handed_digit_3
        self.hand_1, self.hand_2, self.hand_3 = self._hands()
        self.digit_1, self.digit_2, self.digit_3 = self._digits()
        self.hand = self.hand_2

    def __repr__(self):
        self_str = "Trigram {},{},{}:{},{},{}".format(self.midi_1, self.midi_2, self.midi_3,
                                                      self.handed_digit_1, self.handed_digit_2, self.handed_digit_3)
        return self_str


class Ruler(ABC):
    def __init__(self, max_leap=16):
        # The precise semantics of max_leap are defined by derived classes.
        self._max_leap = max_leap

    def distance(self, from_midi, to_midi):
        """
        Estimate the distance between two piano keys identified by MIDI code.
        The original Parncutt paper simply uses semitone differences.
        :param from_midi: The starting piano key.
        :param to_midi: The ending piano key.
        :return: The distance between the two keys.
        """
        calculated_d = to_midi - from_midi
        abs_max_d = self.absolute_max_distance()
        if abs(calculated_d) > abs_max_d:
            if calculated_d < 0:
                return abs_max_d * -1
            else:
                return abs_max_d
        return calculated_d

    def absolute_max_distance(self):
        max_d = self._max_leap + 1
        return max_d


class CacheRuler(Ruler):
    def __init__(self, max_leap=16, preload=False):
        super().__init__(max_leap=max_leap)
        self._preload_cache = preload
        self._cache = {}
        self._distance_method = None
        if preload:
            self.cache_all()
            self._distance_method = self.cached_distance
        else:
            self._distance_method = self.dynamic_distance

    def cache_all(self):
        for from_midi in range(21, 109):
            for to_midi in range(21, 109):
                self._cache[(from_midi, to_midi)] = self.calculated_distance(from_midi, to_midi)

    def add_distance_to_cache(self, from_midi, to_midi, d):
        self._cache[(from_midi, to_midi)] = d

    def cached_distance(self, from_midi, to_midi):
        return self._cache[(from_midi, to_midi)]

    def dynamic_distance(self, from_midi, to_midi):
        if (from_midi, to_midi) in self._cache:
            return self._cache[(from_midi, to_midi)]
        d = self.calculated_distance(from_midi, to_midi)
        self.add_distance_to_cache(from_midi, to_midi, d)
        return self.cached_distance(from_midi, to_midi)

    def distance(self, from_midi, to_midi):
        return self._distance_method(from_midi, to_midi)

    @abstractmethod
    def calculated_distance(self, from_midi, to_midi):
        return 0.0


class PhysicalRuler(Ruler):
    def __init__(self, max_leap=16):
        super().__init__(max_leap=max_leap)
        self._key_positions = PhysicalRuler.horizontal_key_positions()
        self._bounds_for_semitone_interval = None
        self.set_bounds_for_semitone_intervals()

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
        # raise Exception("Distance between {0} and {1} could not be calculated".format(from_midi, to_midi))
        # Following Nakamura's lead, we treat all distances over a certain length the same.
        i += 1
        return multiplier * i

    def set_bounds_for_semitone_intervals(self):
        avg_distances = list()
        for interval_size in range(0, self._max_leap):
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

    @staticmethod
    def horizontal_key_positions():
        """
        Return a dictionary mapping MIDI pitch numbers to the millimeter offsets
        to their lengthwise center lines on the keyboard.
        """
        positions = dict()
        #           A    A#    B  C     C#   D   D#  E     F   F#  G     G#
        offsets = [11.5, 15.5, 8, 23.5, 9.5, 14, 14, 9.5, 23.5, 8, 15.5, 11.5]
        cycle_index = 0
        value = 0
        for midi_id in range(21, 109):
            value += offsets[cycle_index % len(offsets)]
            positions[midi_id] = value
            cycle_index += 1

        return positions


class ImaginaryBlackKeyRuler(CacheRuler):
    def __init__(self, max_leap=16, preload=False):
        super().__init__(max_leap=max_leap, preload=preload)

    def calculated_distance(self, from_midi, to_midi):
        d = 0
        black_to_left = is_black(from_midi)
        left_midi = from_midi
        right_midi = to_midi
        if from_midi > to_midi:
            left_midi = to_midi
            right_midi = from_midi
        for midi in range(left_midi + 1, right_midi + 1):
            if is_white(midi) and not black_to_left:
                d += 1
            black_to_left = is_black(midi)
            d += 1
        if from_midi > to_midi:
            d *= -1
        return d


class Parncutt(D.TrainedDactyler):
    def train(self, d_corpus, staff="both", segregate=True, segmenter=None, annotation_indices=None):
        pass

    def init_rules(self, rules=dict()):
        if rules:
            self._rules = rules
        else:
            self.init_default_rules()

    def init_default_rules(self):
        self._rules = {
            'str': self.assess_stretch,                 # Rule 1 ("Stretch")
            'sma': self.assess_small_span,              # Rule 2 ("Small-Span")
            'lar': self.assess_large_span,              # Rule 3 ("Large-Span")
            'pcc': self.assess_position_change_count,   # Rule 4 ("Position-Change-Count")
            'pcs': self.assess_position_change_size,    # Rule 5 ("Position-Change-Size")
            'wea': self.assess_weak_finger,             # Rule 6 (wea "Weak-Finger")
            '345': self.assess_345,                     # Rule 7 ("Three-Four-Five")
            '3t4': self.assess_3_to_4,                  # Rule 8 ("Three-to-Four")
            'bl4': self.assess_4_on_black,              # Rule 9 ("Four-on-Black")
            'bl1': self.assess_thumb_on_black,          # Rule 10 ("Thumb-on-Black")
            'bl5': self.assess_5_on_black,              # Rule 11 ("Five-on-Black")
            'pa1': self.assess_thumb_passing            # Rule 12 ("Thumb-Passing")
        }

    def init_rule_weights(self, weights=list()):
        self._weights = {}
        if weights:
            if len(weights) != len(self._rules):
                raise Exception("Weights specified do not match all rule names.")
            for rule in self._rules:
                if rule not in weights:
                    raise Exception("Weight for rule tag {} is not specified.".format(rule))
            for rule in self._rules:
                self._weights[rule] = 0
            for rule in weights:
                if rule not in self._rules:
                    raise Exception("Weighting a nonexistent rule: {}".format(rule))
                self._weights[rule] = weights[rule]
        else:
            self.init_default_rule_weights()

    def get_rule_weights(self):
        return self._weights

    def init_default_rule_weights(self):
        for rule in self._rules:
            self._weights[rule] = 1

    def set_rule_weight(self, tag, weight):
        if tag not in self._weights:
            raise Exception("Cannot re-weight unknown rule.")
        self._weights[tag] = weight

    def init_costs(self):
        costs = {}
        for wt in self._weights:
            costs[wt] = 0
        return costs

    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive", ruler=Ruler(),
                 pruning_method='max', finger_spans=FINGER_SPANS, version=(1, 0, 0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner,
                         staff_combiner=staff_combiner, version=version)
        # self._finger_spans = FINGER_SPANS
        # if finger_spans:
        self._finger_spans = finger_spans
        self._ruler = ruler
        self._rules = {}
        self._last_segment_all_paths = None  # Generator of all paths for last segment processed.
        self._pruning_method = None
        self.pruning_method(method=pruning_method)
        self._weights = {}
        self.init_rules()
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

        # print("BAD {0} to {1} trans of span {2} (between {3} and {4})".
              # format(from_digit, to_digit, required_span, min_prac, max_prac))
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

    def distance(self, from_midi, to_midi):
        return self._ruler.distance(from_midi, to_midi)

    def assess_stretch(self, trigram):
        if not trigram.same_hands():
            return 0
        # Rule 1 ("Stretch")
        cost = 0
        if not trigram.midi_1:
            return 0

        semitone_diff_12 = self.distance(trigram.midi_1, trigram.midi_2)
        max_comf_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxComf']
        min_comf_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinComf']

        # "Assign 2 points for each semitone that an interval exceeds MaxComf or is less than MinComf."
        if semitone_diff_12 > max_comf_12:
            cost = 2 * (semitone_diff_12 - max_comf_12)
        elif semitone_diff_12 < min_comf_12:
            cost = 2 * (min_comf_12 - semitone_diff_12)
        return cost

    def assess_small_per_rule(self, trigram):
        if not trigram.same_hands():
            return 0
        # Rule 2 ("Small-Span")
        # "For finger pairs including the thumb, assign 1 point for each semitone that an interval is
        # less than MinRel. For finger pairs not including the thumb, assign 2 points per semitone."
        cost = 0
        if not trigram.midi_1:
            return 0

        semitone_diff_12 = self.distance(trigram.midi_1, trigram.midi_2)
        min_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinRel']

        span_penalty = 2
        if trigram.digit_1 == C.THUMB or trigram.digit_2 == C.THUMB:
            span_penalty = 1
        if semitone_diff_12 < min_rel_12:
            cost = span_penalty * (min_rel_12 - semitone_diff_12)
        return cost

    def assess_small_span(self, trigram, span_penalty=2, thumb_penalty=1):
        if not trigram.same_hands():
            return 0
        # Rule 2 ("Small-Span")
        # "For finger pairs including the thumb, assign 1 point for each semitone that an interval is
        # less than MinRel. For finger pairs not including the thumb, assign 2 points per semitone."
        cost = 0
        if not trigram.midi_1:
            return cost

        semitone_diff = self.distance(trigram.midi_1, trigram.midi_2)
        digit_1 = D.Dactyler.digit_only(trigram.handed_digit_1)
        digit_2 = D.Dactyler.digit_only(trigram.handed_digit_2)
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = thumb_penalty

        hand = D.Dactyler.digit_hand(trigram.handed_digit_1)
        min_rel = None
        max_rel = None
        if hand == '>':
            if digit_1 < digit_2:
                min_rel = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinRel']
            elif digit_1 > digit_2:
                max_rel = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
            else:
                return cost
        else:
            if digit_1 > digit_2:
                min_rel = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinRel']
            elif digit_1 < digit_2:
                max_rel = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
            else:
                return cost

        if min_rel is not None and semitone_diff < min_rel:
            cost = span_penalty * (min_rel - semitone_diff)
        if max_rel is not None and semitone_diff > max_rel:
            cost = span_penalty * (semitone_diff - max_rel)
        return cost

    def assess_small_span_balliauw(self, trigram):
        return self.assess_small_span(trigram=trigram, span_penalty=1)

    def assess_large_span_per_rule(self, trigram, tag='lar'):
        # Rule 3 ("Large-Span")
        # "For finger pairs including the thumb, assign 1 point for each semitone that an interval
        # exceeds MaxRel. For finger pairs not including the thumb, assign 2 points per semitone."
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1:
            return cost

        digit_1 = D.Dactyler.digit_only(trigram.handed_digit_1)
        digit_2 = D.Dactyler.digit_only(trigram.handed_digit_2)
        span_penalty = 2
        if digit_1 == C.THUMB or digit_2 == C.THUMB:
            span_penalty = 1
        semitone_diff_12 = self.distance(trigram.midi_1, trigram.midi_2)
        max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
        if semitone_diff_12 > max_rel_12:
            cost = span_penalty * (semitone_diff_12 - max_rel_12)
        return cost

    def assess_large_span(self, trigram, severe_penalty=2, penalty=1):
        # Rule 3 ("Large-Span") as described in Parncutt text and implied in results reported,
        # NOT as defined in the stated Rule 3.
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1:
            return cost

        absolute_semitone_diff_12 = abs(self.distance(trigram.midi_1, trigram.midi_2))
        span_penalty = severe_penalty
        if trigram.digit_1 == C.THUMB or trigram.digit_2 == C.THUMB:
            span_penalty = penalty

        if trigram.hand == '>':
            if trigram.digit_1 < trigram.digit_2 and trigram.midi_1 < trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
            elif trigram.digit_1 > trigram.digit_2 and trigram.midi_1 > trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_2, trigram.handed_digit_1)]['MaxRel']
            else:
                return cost
        else:
            if trigram.digit_1 < trigram.digit_2 and trigram.midi_1 < trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_2, trigram.handed_digit_1)]['MaxRel']
            elif trigram.digit_1 > trigram.digit_2 and trigram.midi_1 > trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
            else:
                return cost

        if absolute_semitone_diff_12 > max_rel_12:
            cost = span_penalty * (absolute_semitone_diff_12 - max_rel_12)
        return cost

    def assess_large_span_jacobs(self, trigram):
        # Rule 3 ("Large-Span") as described in Parncutt text and implied in results reported,
        # NOT as defined in the stated Rule 3.
        # Modified per Jacobs.
        return self.assess_large_span(trigram=trigram, severe_penalty=1)

    def assess_weak_finger_jacobs(self, trigram, tag='weaj'):
        # Rule 6 (wea "Weak-Finger")
        # Assign 1 point every time finger 4 is used (but no longer finger 5).
        # Per Jacobs.
        cost = 0
        if trigram.digit_2 == C.RING:
            cost = self._weights[tag]
        return cost

    def raw_position_change_count(self, trigram):
        if not trigram.same_hands():
            return 0
        if not trigram.midi_1 or not trigram.midi_3:
            return 0

        pcc = 0
        semitone_diff_13 = self.distance(trigram.midi_1, trigram.midi_3)
        # if semitone_diff_13 != 0:  # FIXME: This is in the code Parncutt shared and needed to reproduce
                                     # results for A and E, but is contradicted by Figure 2(iv)
                                     # example in paper.
        max_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MaxComf']
        min_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MinComf']
        max_prac_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MaxPrac']
        min_prac_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MinPrac']

        if semitone_diff_13 > max_comf_13:
            if trigram.digit_2 == C.THUMB and is_between(trigram.midi_2, trigram.midi_1, trigram.midi_3) \
                    and semitone_diff_13 > max_prac_13:
                pcc = 2  # A "full change"
            else:
                pcc = 1  # A "half change"
        elif semitone_diff_13 < min_comf_13:
            if trigram.digit_2 == C.THUMB and is_between(trigram.midi_2, trigram.midi_1, trigram.midi_3) \
                    and semitone_diff_13 < min_prac_13:
                pcc = 2  # A "full change"
            else:
                pcc = 1  # A "half change"
        return pcc

    def assess_position_change_count(self, trigram):
        # Rule 4 ("Position-Change-Count")
        # "Assign 2 points for every full change of hand position and 1 point for every half change.
        # A change of hand position occurs whenever the first and third notes in a consecutive
        # group of three span an interval that is greater than MaxComf or less than MinComf
        # for the corresponding fingers. In a full change, three conditions are satisfied
        # simultaneously: The finger on the second of the three notes is the thumb; the second pitch
        # lies between the first and third pitches; and the interval between the first and third pitches
        # is greater than MaxPrac or less than MinPrac. All other changes are half changes."
        cost = self.raw_position_change_count(trigram)
        return cost

    def assess_position_change_size(self, trigram):
        # Rule 5 ("Position-Change-Size")
        # "If the interval spanned by the first and third notes in a group of three is less than MinComf,
        # assign the difference between the interval and MinComf (expressed in semitones). Conversely,
        # if the interval is greater than MaxComf, assign the difference between the interval and MaxComf."
        cost = 0
        ### if semitone_diff_13 != 0:  # This is in the code Parncutt shared, but is contradicted in paper.
        if not trigram.midi_1 or not trigram.midi_3:
            return cost

        if not trigram.same_hands():
            return 0

        semitone_diff_13 = self.distance(trigram.midi_1, trigram.midi_3)
        max_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MaxComf']
        min_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MinComf']

        if semitone_diff_13 < min_comf_13:
            cost = (min_comf_13 - semitone_diff_13)
        elif semitone_diff_13 > max_comf_13:
            cost = (semitone_diff_13 - max_comf_13)
        return cost

    def assess_weak_finger(self, trigram):
        # Rule 6 (wea "Weak-Finger")
        # "Assign 1 point every time finger 4 or finger 5 is used."
        if trigram.digit_2 == C.RING or trigram.digit_2 == C.LITTLE:
            return 1
        return 0

    def assess_345(self, trigram):
        # Rule 7 ("Three-Four-Five")
        # "Assign 1 point every time fingers 3, 4, and 5 occur consecutively in any order,
        # even when groups overlap."

        if not trigram.same_hands():
            return 0

        cost = 0
        finger_hash = {
            trigram.digit_1: True,
            trigram.digit_2: True,
            trigram.digit_3: True
        }
        if C.MIDDLE in finger_hash and C.RING in finger_hash and C.LITTLE in finger_hash:
            cost = 1
        return cost

    def assess_3_to_4(self, trigram):
        # Rule 8 ("Three-to-Four")
        # "Assign 1 point each time finger 3 is immediately followed by finger 4."
        if not trigram.same_hands():
            return 0
        if trigram.digit_1 == C.MIDDLE and trigram.digit_2 == C.RING:
            return 1
        return 0

    def assess_4_on_black(self, trigram):
        # Rule 9 ("Four-on-Black")
        # "Assign 1 point each time fingers 3 and 4 occur consecutively in any order with 3 on
        # white and 4 on black."
        if (trigram.digit_1 == C.RING and is_black(trigram.midi_1) and trigram.digit_2 == C.MIDDLE and is_white(trigram.midi_2)) or \
                (trigram.digit_1 == C.MIDDLE and is_white(trigram.midi_1) and trigram.digit_2 == C.RING and is_black(trigram.midi_2)):
            return 1
        return 0

    def assess_thumb_on_black(self, trigram):
        # Rule 10 ("Thumb-on-Black")
        # "Assign 1 point whenever the thumb plays a black key."
        cost = 0
        if trigram.digit_2 != C.THUMB or is_white(trigram.midi_2):
            return cost

        cost += 1

        # "If the immediately preceding note is white, assign a further 2 points."
        if trigram.digit_1 and trigram.digit_2 == C.THUMB and is_black(trigram.midi_2) and is_white(trigram.midi_1):
            cost += 2

        # "If the immediately following note is white, assign a further 2 points."
        if trigram.digit_3 and trigram.digit_2 == C.THUMB and is_black(trigram.midi_2) and is_white(trigram.midi_3):
            cost += 2
        return cost

    def assess_5_on_black(self, trigram):
        # Rule 11 ("Five-on-Black")
        # "If the fifth finger plays a black key and the immediately preceding and following notes
        # are also black, assign 0 points. If the immediately preceding note is white, assign 2 points.
        # If the immediately following key is white, assign 2 further points."
        cost = 0
        if trigram.digit_2 == C.LITTLE and is_black(trigram.midi_2):
            if trigram.midi_1 and is_black(trigram.midi_1) and trigram.midi_3 and is_black(trigram.midi_3):
                cost = 0
            else:
                if trigram.midi_1 and is_white(trigram.midi_1):
                    cost = 2
                if trigram.midi_3 and is_white(trigram.midi_3):
                    cost += 2
        return cost

    def assess_thumb_passing(self, trigram, bad_level_change_cost=3):
        # Rule 12 ("Thumb-Passing")
        # "Assign 1 point for each thumb- or finger-pass on the same level (from white to white
        # or black to black). Assign 3 points if the lower note is white, played by a finger
        # other than the thumb, and the upper is black, played by the thumb." Invert logic for
        # the left hand. Passing (pivoting) with the thumb on white and the finger on black
        # is not penalized. The cost of 3 is configurable.
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.digit_1:
            return cost

        if trigram.hand == '>':
            if trigram.digit_1 == C.THUMB and trigram.midi_2 < trigram.midi_1:  # Finger crossing over thumb, descending.
                if (is_white(trigram.midi_1) and is_white(trigram.midi_2)) or (is_black(trigram.midi_1) and is_black(trigram.midi_2)):
                    cost = 1  # Same level.
                elif is_black(trigram.midi_1):
                    cost = bad_level_change_cost
            if trigram.digit_2 == C.THUMB and trigram.midi_2 > trigram.midi_1:  # Thumb passing under finger, ascending.
                if (is_white(trigram.midi_1) and is_white(trigram.midi_2)) or (is_black(trigram.midi_1) and is_black(trigram.midi_2)):
                    cost = 1
                elif is_black(trigram.midi_2):
                    cost = bad_level_change_cost
        else:
            if trigram.digit_1 == C.THUMB and trigram.midi_2 > trigram.midi_1:  # Finger crossing over thumb, ascending.
                if (is_white(trigram.midi_1) and is_white(trigram.midi_2)) or (is_black(trigram.midi_1) and is_black(trigram.midi_2)):
                    cost = 1
                elif is_black(trigram.midi_1):
                    cost = bad_level_change_cost
            if trigram.digit_2 == C.THUMB and trigram.midi_2 < trigram.midi_1:  # Thumb passing under finger, descending.
                if (is_white(trigram.midi_1) and is_white(trigram.midi_2)) or (is_black(trigram.midi_1) and is_black(trigram.midi_2)):
                    cost = 1
                elif is_black(trigram.midi_2):
                    cost = bad_level_change_cost
        return cost

    def assess_thumb_passing_balliauw(self, trigram):
        return self.assess_thumb_passing(trigram=trigram, bad_level_change_cost=2)

    def assess_large_span_badgerow(self, trigram):
        # Rule 3 ("Large-Span") as described in Parncutt text and implied in results reported,
        # NOT as defined in the stated Rule 3. Amended as suggested by Badgerow:
        #
        # "If PCC (Position Change Count) is less than or equal to 1, assign points exceeding
        # MaxComf, not MaxRel. So, for finger pairs including the thumb, assign 1 point for
        # each semitone that an interval exceeds MaxComf. For finger pairs not including
        # the thumb, assign 2 points per semitone than [that] an interval exceeds MaxComf."
        #
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1:
            return cost

        absolute_semitone_diff_12 = abs(self.distance(trigram.midi_1, trigram.midi_2))
        span_penalty = 2
        if trigram.digit_1 == C.THUMB or trigram.digit_2 == C.THUMB:
            span_penalty = 1

        hand = D.Dactyler.digit_hand(trigram.handed_digit_1)
        if hand == '>':
            if trigram.digit_1 < trigram.digit_2 and trigram.midi_1 < trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
                max_comf_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxComf']
            elif trigram.digit_1 > trigram.digit_2 and trigram.midi_1 > trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_2, trigram.handed_digit_1)]['MaxRel']
                max_comf_12 = self._finger_spans[(trigram.handed_digit_2, trigram.handed_digit_1)]['MaxComf']
            else:
                return 0
        else:
            if trigram.digit_1 < trigram.digit_2 and trigram.midi_1 < trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_2, trigram.handed_digit_1)]['MaxRel']
                max_comf_12 = self._finger_spans[(trigram.handed_digit_2, trigram.handed_digit_1)]['MaxComf']
            elif trigram.digit_1 > trigram.digit_2 and trigram.midi_1 > trigram.midi_2:
                max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
                max_comf_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxComf']
            else:
                return 0

        raw_pcc = 2
        if trigram.midi_3:
            raw_pcc = self.raw_position_change_count(trigram)
        if raw_pcc <= 1:
            if absolute_semitone_diff_12 > max_comf_12:
                cost = span_penalty * (absolute_semitone_diff_12 - max_comf_12)
        else:
            if absolute_semitone_diff_12 > max_rel_12:
                cost = span_penalty * (absolute_semitone_diff_12 - max_rel_12)
        return cost

    def assess_alternation_pairing_badgerow(self, trigram):
        # New Rule ("Alternation-Pairing") from Justin Badgerow
        # "Assign 1 point for 3-4-3 or 4-3-4 combinations and 1 point for 4-5-4 or 5-4-5 combinations."
        ### FIXME: Justin says, "MAYBE DELETE??"
        if not trigram.same_hands():
            return 0
        cost = 0
        if (trigram.digit_1 == 3 and trigram.digit_2 == 4 and trigram.digit_3 == 3) or \
                (trigram.digit_1 == 4 and trigram.digit_2 == 3 and trigram.digit_3 == 4) or \
                (trigram.digit_1 == 4 and trigram.digit_2 == 5 and trigram.digit_3 == 4) or \
                (trigram.digit_1 == 5 and trigram.digit_2 == 4 and trigram.digit_3 == 5):
            cost = 1
        return cost

    def assess_alternation_finger_change_badgerow(self, trigram):
        # New Rule ("Alternation-Finger-Change") from Justin Badgerow
        # "On three note passages where the 1st and 3rd note are the same, add a 1 point
        # deduction when a different finger is on the 1st and 3rd pitch.
        if not trigram.same_hands():
            return 0
        cost = 0
        if trigram.digit_1 and trigram.digit_3 and trigram.midi_1 == trigram.midi_3 and \
                trigram.digit_1 != trigram.digit_3:
            cost = 1
        return cost

    def assess_black_thumb_pivot_badgerow(self, trigram):
        if not trigram.same_hands():
            return 0
        cost = 0
        if trigram.midi_1 == trigram.midi_2:
            return cost
        if not trigram.midi_1:
            return cost
        if trigram.midi_1 == trigram.midi_2:
            return cost
        if trigram.hand_1 != trigram.hand_2:
            return cost

        if trigram.hand_1 == '>':
            if trigram.midi_2 < trigram.midi_1:  # descending
                if trigram.digit_1 == C.THUMB and is_black(trigram.midi_1) and trigram.digit_2 in (C.RING, C.LITTLE) and is_white(trigram.midi_2):
                    cost += (trigram.digit_2 - 1)
            else:  # ascending
                if trigram.digit_2 == C.THUMB and is_black(trigram.midi_2) and trigram.digit_1 in (C.RING, C.LITTLE) and is_white(trigram.midi_1):
                    cost += (trigram.digit_2 - 1)
        else:  # LH
            if trigram.midi_2 > trigram.midi_1:  # ascending
                if trigram.digit_1 == C.THUMB and is_black(trigram.midi_1) and trigram.digit_2 in (C.RING, C.LITTLE) and is_white(trigram.midi_2):
                    cost += (trigram.digit_2 - 1)
            else:  # descending
                if trigram.digit_2 == C.THUMB and is_black(trigram.midi_2) and trigram.digit_1 in (C.RING, C.LITTLE) and is_white(trigram.midi_1):
                    cost += (trigram.digit_2 - 1)
        return cost

    def assess_thumb_on_black_to_weak_badgerow(self, trigram):
        if not trigram.same_hands():
            return 0
        cost = 0
        if trigram.midi_1 == trigram.midi_2:
            return cost
        if trigram.digit_1 != C.THUMB or is_white(trigram.midi_1):
            return cost
        # So a thumb is playing a black note 1.

        if is_black(trigram.midi_2) or trigram.digit_2 not in (C.RING, C.LITTLE):
            return cost
        # So a weak finger is playing a white note 2.

        if trigram.hand_1 != trigram.hand_2:
            return cost

        # For RH, ANY descending move from 1-4 or 1-5 from black to white key, regardless of size,
        # will be penalized an extra 3 points for 1-4 pairs and an extra 4 points for 1-5 pairs.
        # Assess the same penalties for ascending intervals, if said intervals are less
        # than MinRel. And flip this around for the left hand.

        if trigram.hand_1 == '>':
            if trigram.midi_1 > trigram.midi_2:  # descending
                cost += (trigram.digit_2 - 1)
            else:
                distance = self.distance(trigram.midi_1, trigram.midi_2)
                min_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinRel']
                if distance < min_rel_12:
                    cost += (trigram.digit_2 - 1)
        else:  # Left hand
            if trigram.midi_2 > trigram.midi_1:  # ascending
                cost += (trigram.digit_2 - 1)
            else:
                distance = self.distance(trigram.midi_1, trigram.midi_2)
                min_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinRel']
                if distance < min_rel_12:
                    cost += (trigram.digit_2 - 1)
        return cost

    def assess_weak_to_thumb_on_black_badgerow(self, trigram):
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1 or trigram.midi_1 == trigram.midi_2:
            return cost
        if trigram.digit_2 != C.THUMB or is_white(trigram.midi_2):
            return cost
        # So thumb is playing a black note 2.

        if is_black(trigram.midi_1) or trigram.digit_1 not in (C.RING, C.LITTLE):
            return cost
        # So a weak finger is playing a white note 1.

        if trigram.hand_1 != trigram.hand_2:
            return cost

        # For the RH, any ascending move from 4-1 or 5-1, from black to white key, regardless of size,
        # assess 3 points for 4-1 pairs and 4 points for 5-1 pairs.
        # Assess the same penalties for descending intervals, if said intervals are more than MaxRel.
        if trigram.hand_1 == '>':
            if trigram.midi_2 > trigram.midi_1:  # ascending
                cost += (trigram.digit_1 - 1)
            else:
                distance = self.distance(trigram.midi_1, trigram.midi_2)
                max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
                if distance > max_rel_12:
                    cost += (trigram.digit_1 - 1)
        else:  # Left hand
            if trigram.midi_2 > trigram.midi_1:  # ascending
                distance = self.distance(trigram.midi_1, trigram.midi_2)
                max_rel_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxRel']
                if distance > max_rel_12:
                    cost += (trigram.digit_1 - 1)
            else:
                cost += (trigram.digit_1 - 1)
        return cost

    # def assess_thumb_on_black(self, trigram):
    #     # Rule 10 ("Thumb-on-Black")
    #     # "Assign 1 point whenever the thumb plays a black key."
    #     if trigram.digit_2 != C.THUMB or is_white(trigram.midi_2):
    #         return
    #
    #     self._costs['bl1'] += self._weights['bl1']
    #
    #     # "If the immediately preceding note is white, assign a further 2 points."
    #     if trigram.digit_1 and trigram.digit_2 == C.THUMB and is_black(trigram.midi_2) and is_white(trigram.midi_1):
    #         self._costs['bl1'] += 2 * self._weights['bl1']
    #
    #     # "If the immediately following note is white, assign a further 2 points."
    #     if trigram.digit_3 and trigram.digit_2 == C.THUMB and is_black(trigram.midi_2) and is_white(trigram.midi_3):
    #         self._costs['bl1'] += 2 * self._weights['bl1']
    #
    #         # Justin's amendment: "When the thumb plays a black key, if the preceding OR
    #         # following note is finger 5 on a white key, assign a further 2 points for each usage."
    #         if trigram.digit_1 == C.LITTLE and is_white(trigram.midi_1):
    #             self._costs['bl1'] += 2 * self._weights['bl1']
    #         if trigram.digit_3 == C.LITTLE and is_white(trigram.midi_3):
    #             self._costs['bl1'] += 2 * self._weights['bl1']

    def assess_position_change_balliauw(self, trigram):
        # Balliauw Rule 3: "For three consecutive notes: If the distance between a first and third note is
        # below MinComf or exceeds MaxComf: add one point. In addition to that, if the pitch of the second
        # note is between the other two pitches, is played by the thumb and the distance between the first and
        # third note is below MinPrac or exceeds MaxPrac: add another point. Finally, if the first and third note
        # have the same pitch, but are played by a different finger: add another point."
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1 or not trigram.midi_3:
            return cost

        semitone_diff_13 = self.distance(trigram.midi_1, trigram.midi_3)
        max_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MaxComf']
        min_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MinComf']
        max_prac_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MaxPrac']
        min_prac_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MinPrac']

        if semitone_diff_13 < min_comf_13 or semitone_diff_13 > max_comf_13:
            cost += 1
            if (trigram.midi_1 < trigram.midi_2 < trigram.midi_3) and trigram.digit_2 == C.THUMB and \
                    (semitone_diff_13 < min_prac_13 or semitone_diff_13 > max_prac_13):
                cost += 1
        if trigram.midi_1 == trigram.midi_3 and trigram.handed_digit_1 != trigram.handed_digit_3:
            cost += 1
        return cost

    def assess_position_comfort_balliauw(self, trigram):
        # Balliauw Rule 4: "For every unit the distance between a first and third note is below MinComf
        # or exceeds MaxComf."
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1 or not trigram.midi_3:
            return cost

        semitone_diff_13 = self.distance(trigram.midi_1, trigram.midi_3)
        max_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MaxComf']
        min_comf_13 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_3)]['MinComf']
        if semitone_diff_13 > max_comf_13:
            cost = (semitone_diff_13 - max_comf_13)
        elif semitone_diff_13 < min_comf_13:
            cost = (max_comf_13 - semitone_diff_13)
        return cost

    def assess_repeat_finger_on_position_change_balliauw(self, trigram):
        # Balliauw Rule 12: "For a different first and third consecutive note, played by the same finger,
        # and the second pitch being the middle one." This is clarified this way: "Rule 12 prevents the repetitive
        # use of a finger in combination with a hand position change. For instance, a sequenceC4–G4–C5fingered 2–1–2
        # in the right hand forces the pianist to reuse the second finger very quickly" (p.516).
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1 or not trigram.midi_3:
            return cost
        if trigram.midi_1 != trigram.midi_3 and trigram.handed_digit_1 == trigram.handed_digit_3 and \
                self.raw_position_change_count(trigram) > 0:
            cost = 1
        return cost

    def assess_impractical_balliauw(self, trigram, penalty=10):
        # Balliauw Rule 13: "For every unit where the distance between two following notes
        # is below MinPrac or exceeds MaxPrac." Penalize +10.
        if not trigram.same_hands():
            return 0
        cost = 0
        if not trigram.midi_1:
            return 0

        semitone_diff_12 = self.distance(trigram.midi_1, trigram.midi_2)
        max_prac_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MaxPrac']
        min_prac_12 = self._finger_spans[(trigram.handed_digit_1, trigram.handed_digit_2)]['MinPrac']

        # "Assign 2 points for each semitone that an interval exceeds MaxComf or is less than MinComf."
        if semitone_diff_12 > max_prac_12:
            cost = penalty * (semitone_diff_12 - max_prac_12)
        elif semitone_diff_12 < min_prac_12:
            cost = penalty * (min_prac_12 - semitone_diff_12)
        return cost

    def assess_nonrepeating_finger_balliauw(self, trigram):
        # Balliauw Rule 15: "For consecutive slices containing exactly the same notes (with identical pitches),
        # played by a different finger, for each different finger." For our purposes, a slice is a note.
        cost = 0
        if not trigram.midi_1:
            return 0
        if trigram.midi_1 == trigram.midi_2 and trigram.handed_finger_1 != trigram.handed_finger_2:
            cost = 1
        return cost

    def rules(self):
        return self._rules

    def trigram_node_cost(self, trigram_node):
        """
        Determine the cost associated with a trigram node configured as input.
        :param trigram_node: The TrigramNode under consideration.
        :return: cost, costs: The total (scalar integer) cost associated with the node, and a dictionary
        detailing the specific subcosts contributing to the total.
        """
        cost = 0
        costs = self.init_costs()

        for tag, rule_method in self._rules.items():
            raw_cost = rule_method(trigram_node)
            weighted_cost = self._weights[tag] * raw_cost
            costs[tag] = weighted_cost
            cost += weighted_cost
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

            trigram = TrigramNode(midi_1=midi_1, handed_digit_1=hd_1,
                                  midi_2=midi_2, handed_digit_2=hd_2,
                                  midi_3=midi_3, handed_digit_3=hd_3)
            cost, detail = self.trigram_node_cost(trigram)
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
                                trigram_node = TrigramNode(midi_1=midi_1, handed_digit_1=digit_1,
                                                           midi_2=midi_2, handed_digit_2=digit_2,
                                                           midi_3=midi_3, handed_digit_3=digit_3)
                                weight, weights = self.trigram_node_cost(trigram_node)
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
        :param g: The weighted tri-node graph. Weights must be specified via a "weight" edge parameter. Fingerings
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
            suggest_map = dict()
            suggestions = list()
            costs = list()
            details = list()
            k_best_paths = list(islice(nx.shortest_simple_paths(
                g, source=0, target=target_id, weight="weight"), k))
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
                if segment_abcdf in suggest_map:
                    suggest_map[segment_abcdf] += 1
                else:
                    suggest_map[segment_abcdf] = 1
                costs.append(suggestion_cost)
                details.append(rule_costs)

            # print("TOTAL: {0} DISTINCT: {1} COSTS: {2}".format(len(suggestions), len(suggest_map), costs))
            return suggestions, costs, details

    def generate_segment_advice(self, segment, staff, offset=0, cycle=None,
                                handed_first_digit=None, handed_last_digit=None, k=None):
        """
        Generate a set of k ranked fingering suggestions for the given segment.
        :param segment: The segment to work with, as a music21 score m21_object.
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
    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive", ruler=PhysicalRuler(),
                 pruning_method='max', finger_spans=FINGER_SPANS, version=(1, 0, 0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner, ruler=ruler,
                         staff_combiner=staff_combiner, pruning_method=pruning_method,
                         finger_spans=finger_spans, version=version)

    def init_default_rules(self):
        # Rule 1 ("Stretch")
        # Rule 2 ("Small-Span")
        # Rule 3 (larj "Large-Span")
        # Rule 4 ("Position-Change-Count")
        # Rule 5 ("Position-Change-Size")
        # Rule 6 (weaj "Weak-Finger")
        # Rule 7 ("Three-Four-Five") NOT DONE in the interests of parsimony.
        # Rule 8 ("Three-to-Four")
        # Rule 9 ("Four-on-Black")
        # Rule 10 ("Thumb-on-Black")
        # Rule 11 ("Five-on-Black")
        # Rule 12 ("Thumb-Passing")
        self._rules = {
            'str': self.assess_stretch,
            'sma': self.assess_small_span,
            'larj': self.assess_large_span_jacobs,
            'pcc': self.assess_position_change_count,
            'pcs': self.assess_position_change_size,
            'weaj': self.assess_weak_finger_jacobs,
            # '345': self.assess_345
            '3t4': self.assess_3_to_4,
            'bl4': self.assess_4_on_black,
            'bl1': self.assess_thumb_on_black,
            'bl5': self.assess_5_on_black,
            'pa1': self.assess_thumb_passing
        }


class Badgerow(Parncutt):
    def init_default_rules(self):
        # Rule 1 ("Stretch")
        # Rule 2 ("Small-Span")
        # Rule 3 (larb "Large-Span")
        # Rule 4 ("Position-Change-Count")
        # Rule 5 ("Position-Change-Size")
        # Rule 6 (wea "Weak-Finger")
        # Rule 7 ("Three-Four-Five")
        # Rule 8 ("Three-to-Four")
        # Rule 9 ("Four-on-Black")
        # Rule 10 ("Thumb-on-Black")
        # Rule 11 ("Five-on-Black")
        # Rule 12 ("Thumb-Passing")
        # New rule (aprb "Alternation-Pairing")
        # self.assess_alternation_pairing_badgerow(trigram_node)
        # New rule (afcb "Alternation-Finger-Change")
        # self.assess_alternation_finger_change_badgerow(costs, trigram_node)
        # New rule (b1wb "Thumb-on-Black-to-Weak")
        # self.assess_thumb_on_black_to_weak_badgerow()
        # New rule (wb1b "Weak-to-Thumb-on-Black")
        # self.assess_weak_to_thumb_on_black_badgerow()
        # New rule (b1pb "Black-Thumb-Pivot")
        # self.assess_black_thumb_pivot_badgerow()
        self._rules = {
            'str': self.assess_stretch,
            'sma': self.assess_small_span,
            'larb': self.assess_large_span_badgerow,
            'pcc': self.assess_position_change_count,
            'pcs': self.assess_position_change_size,
            'wea': self.assess_weak_finger,
            '345': self.assess_345,
            '3t4': self.assess_3_to_4,
            'bl4': self.assess_4_on_black,
            'bl1': self.assess_thumb_on_black,
            'bl5': self.assess_5_on_black,
            'pa1': self.assess_thumb_passing,
            'aprb': self.assess_alternation_pairing_badgerow,
            'afcb': self.assess_alternation_finger_change_badgerow,
            'b1wb': self.assess_thumb_on_black_to_weak_badgerow,
            # 'wb1b': self.assess_weak_to_thumb_on_black_badgerow,
            'b1pb': self.assess_black_thumb_pivot_badgerow
        }

    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive", ruler=Ruler(),
                 pruning_method='max', finger_spans=BADGEROW_FINGER_SPANS, version=(1, 0, 0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner, ruler=ruler,
                         staff_combiner=staff_combiner, pruning_method=pruning_method,
                         finger_spans=finger_spans, version=version)


class Balliauw(Parncutt):
    def __init__(self, segmenter=None, segment_combiner="normal", staff_combiner="naive", ruler=ImaginaryBlackKeyRuler(),
                 pruning_method='max', finger_spans=BALLIAUW_LARGE_FINGER_SPANS, version=(1, 0, 0)):
        super().__init__(segmenter=segmenter, segment_combiner=segment_combiner, ruler=ruler,
                         staff_combiner=staff_combiner, pruning_method=pruning_method,
                         finger_spans=finger_spans, version=version)

    def init_default_rule_weights(self):
        super().init_default_rule_weights()
        self._weights['bl1'] = 0.5
        self._weights['bl5'] = 0.5

    def init_default_rules(self, rules=dict()):
        self._rules = {
            'str': self.assess_stretch,                       # Balliauw 1 == Parncutt 1
            'sma': self.assess_small_span_balliauw,           # Balliauw 2
            'pchw': self.assess_position_change_balliauw,     # Balliauw 3
            'pcow': self.assess_position_comfort_balliauw,    # Balliauw 4
            'weaj': self.assess_weak_finger_jacobs,           # Balliauw 5 == Jacobs 6
            '3t4': self.assess_3_to_4,                        # Balliauw 6 == Parncutt 8
            'bl4': self.assess_4_on_black,                    # Balliauw 7 == Parncutt 9
            'bl1': self.assess_thumb_on_black,                # 8 == Parncutt 10, WEIGHTED 0.5
            'bl5': self.assess_5_on_black,                    # 9 == Parncutt 11, WEIGHTED 0.5
            'pa1w': self.assess_thumb_passing,                # 10 and 11 = modified Parncutt 12
            'rfrw': self.assess_repeat_finger_on_position_change_balliauw,  # Balliauw 12
            'impw': self.assess_impractical_balliauw,         # Balliauw 13
            'nrfw': self.assess_nonrepeating_finger_balliauw  # Balliauw 15 (14 is for polyphonic)
        }
