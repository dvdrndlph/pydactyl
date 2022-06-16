__author__ = 'David Randolph'
# Copyright (c) 2018 David A. Randolph.
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


class DSegmenter(ABC):
    """Base class for all phrase-segmentation algorithms."""
    def __init__(self):
        return

    @staticmethod
    def non_default_hand_count(hsd_seq, staff="upper"):
        """
        Returns a count of fingerings that employ a non-default hand.
        """
        non_default_hand = '<'
        if staff == 'lower':
            nondefault_hand = '>'
        bad_hand_cnt = 0
        for fingering in hsd_seq:
            if fingering[0] == non_default_hand:
                bad_hand_cnt += 1
        return bad_hand_cnt

    @staticmethod
    def has_wildcard(hsd_seq):
        for fingering in hsd_seq:
            if fingering[0] == 'x':
                return True
        return False

    @abstractmethod
    def segment_to_orderly_streams(self, d_part, offset=0):
        return

    @abstractmethod
    def segment_to_ordered_offset_notes(self, d_part, offset=0):
        return
