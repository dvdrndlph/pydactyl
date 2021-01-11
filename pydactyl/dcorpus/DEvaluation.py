__author__ = 'David Randolph'
# Copyright (c) 2014-2020 David A. Randolph.
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

from music21 import note, chord
from music21.articulations import Fingering
from pydactyl.dcorpus.PianoFingering import PianoFingering

class DEvaluation:
    """
    Class to determine how well each of the top ranked outputs of a model
    system perform against a given gold-standard human.
    """
    def __init__(self, human_score=None, system_scores=[], staff="both"):
        self._staff = staff
        self._human_score = None
        self._human_note_stream = None
        self.human_score(d_score=human_score)
        self._system_scores = []
        self._system_note_streams = []
        self.system_scores(system_scores)

    @staticmethod
    def _long_short_delta(one_pf, other_note):
        return 0

    def human_score(self, d_score=None):
        if d_score:
            self._human_score = d_score
            self._human_note_stream = d_score.orderly_note_stream(staff=self._staff)
        return self._human_score

    def system_scores(self, system_scores=[]):
        if system_scores:
            for d_score in system_scores:
                self.append_system_score(d_score)
        return self._system_scores

    def append_system_score(self, d_score):
        if d_score:
            self._system_scores.append(d_score)
            stream = d_score.orderly_note_stream(staff=self._staff)
            self._system_note_streams.append(stream)
            return True
        return False

    @staticmethod
    def _delta(one_note, other_note):
        one_pf = PianoFingering.fingering(one_note)
        other_pf = PianoFingering.fingering(other_note)
        if one_pf.strike_digit() != other_pf.strike_digit():
            return 1
        elif one_pf.strike_hand() != other_pf.strike_hand():
            return 1
        return 0

    @staticmethod
    def hamming(one_stream, other_stream):
        if len(one_stream) != len(other_stream):
            raise Exception("Mismatched orderly note streams")
        distance = 0
        for i in range(len(one_stream)):
            distance += DEvaluation._delta(one_stream[i], other_stream[i])
        return distance

    def hamming_at_rank(self, rank):
        index = rank - 1
        distance = DEvaluation.hamming(self._human_note_stream, self._system_note_streams[index])
        return distance

    def normalized_hamming_at_rank(self, rank):
        big_n = self._human_score.note_count(staff=self._staff)
        normed_distance = self.hamming_at_rank(rank=rank) / big_n
        return normed_distance

    @staticmethod
    def pivot_clashes_at_rank(self, rank, decay_function=None):
        index = rank - 1
        one_stream = self._human_note_stream
        other_stream = self._system_note_stream[index]
        return 0
