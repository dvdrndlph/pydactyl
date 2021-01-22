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
    def _pivot(pf_a, pf_b, direction):
        if direction == 0:
            return False
        if pf_a.strike_hand() != pf_b.strike_hand():
            return False # No pivot with different hands involved.
        if direction > 0: # Ascending
            if pf_b.strike_digit() != 1:
                return False
            if pf_a.strike_digit() != pf_b.strike_digit():
                return True
        else: # Descending
            if pf_a.strike_digit() != 1:
                return False
            if pf_a.strike_digit() != pf_b.strike_digit():
                return True
        return False

    @staticmethod
    def _pivot_clash(pf_a, pf_b, pf_x, pf_y, direction):
        is_one_pivot = DEvaluation._pivot(pf_a, pf_b, direction)
        is_other_pivot = DEvaluation._pivot(pf_x, pf_y, direction)
        if is_one_pivot != is_other_pivot:
            return True
        return False

    @staticmethod
    def _get_basic_bigram_info(one_stream, index):
        if index >= len(one_stream):
            raise Exception("Stream index is out of range")

        a_pf = PianoFingering.fingering(one_stream[index])
        b_pf = PianoFingering.fingering(one_stream[index + 1])
        this_note = one_stream[index]
        next_note = one_stream[index + 1]
        this_midi = this_note.pitch.midi
        next_midi = next_note.pitch.midi
        direction = next_midi - this_midi
        return a_pf, b_pf, direction

    def pivot_clashes_at_rank(self, rank, decay_function=None):
        index = rank - 1
        one_stream = self._human_note_stream
        other_stream = self._system_note_streams[index]
        big_n = len(one_stream)

        normalizing_factor = 1
        if decay_function:
            denom = 0
            for n in range(big_n):
                n += 1
                denom += decay_function(big_n=big_n, n=n)
            if denom == 0:
                raise Exception("Bad decay function specified")
            normalizing_factor = big_n / denom

        clashes = 0.0
        for i in range(big_n - 1): # the last note position cannot have a clash.
            a_pf, b_pf, direction = DEvaluation._get_basic_bigram_info(one_stream, i)
            x_pf, y_pf, direction = DEvaluation._get_basic_bigram_info(other_stream, i)
            is_clash = DEvaluation._pivot_clash(a_pf, b_pf, x_pf, y_pf, direction)
            if is_clash:
                if decay_function:
                    clashes += decay_function(big_n=big_n, n=i+1)
                else:
                    clashes += 1
        normalized_clashes = normalizing_factor * clashes
        return normalized_clashes

    @staticmethod
    def _count_pivots_in_fingered_stream(one_stream):
        big_n = len(one_stream)
        pivot_count = 0
        for i in range(big_n - 1): # the last note position cannot have a clash.
            pf_a, pf_b, direction = DEvaluation._get_basic_bigram_info(one_stream, i)
            if DEvaluation._pivot(pf_a, pf_b, direction):
                pivot_count += 1
        return pivot_count

    def print_pivot_count_report(self):
        human_stream = self._human_note_stream
        human_count = DEvaluation._count_pivots_in_fingered_stream(human_stream)
        print("PIVOT COUNTS")
        print("Human: {}".format(human_count))
        for i in range(len(self._system_note_streams)):
            system_stream = self._system_note_streams[i]
            pivot_count = DEvaluation._count_pivots_in_fingered_stream(system_stream)
            print("Sys {}: {}".format(i + 1, human_count))
        print("\n")
