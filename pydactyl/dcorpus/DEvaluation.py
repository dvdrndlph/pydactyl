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

import math
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
    def long_short_delta(one_note, other_note):
        one_pf = PianoFingering.fingering(one_note)
        other_pf = PianoFingering.fingering(other_note)
        if one_pf.strike_hand() != other_pf.strike_hand():
            return 1.0
        one_digit = one_pf.strike_digit()
        other_digit = other_pf.strike_digit()
        if one_digit == other_digit:
            return 0.0
        if (one_digit in (2, 3) and other_digit in (2, 3)) or \
           (one_digit in (3, 4) and other_digit in (3, 4)):
            return 3.0/4.0
        return 0

    @staticmethod
    def hamming_delta(one_note, other_note):
        one_pf = PianoFingering.fingering(one_note)
        other_pf = PianoFingering.fingering(other_note)
        if one_pf.strike_digit() != other_pf.strike_digit():
            return 1
        elif one_pf.strike_hand() != other_pf.strike_hand():
            return 1
        return 0

    def hamming_at_rank(self, rank):
        distance = self.big_delta_at_rank(rank=rank)
        return distance

    def normalized_hamming_at_rank(self, rank):
        big_n = self._human_score.note_count(staff=self._staff)
        normed_distance = self.hamming_at_rank(rank=rank) / big_n
        return normed_distance

    def uniform_decay(big_n, n):
        return big_n - n + 1

    def no_decay(big_n, n):
        return 1

    def big_delta_at_rank(self, rank, delta_function=None, decay_function=None):
        if delta_function is None:
            delta_function = DEvaluation.hamming_delta

        index = rank - 1
        human_stream = self._human_note_stream
        system_stream = self._system_note_streams[index]
        if len(human_stream) != len(system_stream):
            raise Exception("Mismatched orderly note streams")

        big_n = self._human_score.note_count(staff=self._staff)
        normalizing_factor = DEvaluation._normalizing_factor(big_n=big_n, decay_function=decay_function)
        distance = 0
        for i in range(big_n):
            decay_weight = 1
            if decay_function:
                decay_weight = decay_function(big_n=big_n, n=i+1)
            delta_value = delta_function(one_note=human_stream[i], other_note=system_stream[i])
            distance += decay_weight * delta_value
        big_delta = normalizing_factor * distance
        return big_delta

    @staticmethod
    def _is_pivot(pf_a, pf_b, direction):
        if direction == 0:
            return False
        if pf_a.strike_hand() != pf_b.strike_hand():
            return False  # No pivot with different hands involved.
        if direction > 0:  # Ascending
            if pf_b.strike_digit() != 1:
                return False
            if pf_a.strike_digit() != pf_b.strike_digit():
                return True
        else:  # Descending
            if pf_a.strike_digit() != 1:
                return False
            if pf_a.strike_digit() != pf_b.strike_digit():
                return True
        return False

    @staticmethod
    def _is_pivot_clash(pf_a, pf_b, pf_x, pf_y, direction):
        is_one_pivot = DEvaluation._is_pivot(pf_a, pf_b, direction)
        is_other_pivot = DEvaluation._is_pivot(pf_x, pf_y, direction)
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

    @staticmethod
    def _normalizing_factor(big_n, decay_function):
        normalizing_factor = 1
        if decay_function:
            denom = 0
            for n in range(big_n):
                n += 1
                denom += decay_function(big_n=big_n, n=n)
            if denom == 0:
                raise Exception("Bad decay function specified")
            normalizing_factor = big_n / denom
        return normalizing_factor

    def pivot_clashes_at_rank(self, rank, decay_function=None):
        index = rank - 1
        one_stream = self._human_note_stream
        other_stream = self._system_note_streams[index]
        big_n = len(one_stream)
        normalizing_factor = DEvaluation._normalizing_factor(big_n=big_n, decay_function=decay_function)

        clashes = 0.0
        for i in range(big_n - 1): # the last note position cannot have a clash.
            a_pf, b_pf, direction = DEvaluation._get_basic_bigram_info(one_stream, i)
            x_pf, y_pf, direction = DEvaluation._get_basic_bigram_info(other_stream, i)
            is_clash = DEvaluation._is_pivot_clash(a_pf, b_pf, x_pf, y_pf, direction)
            if is_clash:
                if decay_function:
                    clashes += decay_function(big_n=big_n, n=i+1)
                else:
                    clashes += 1
        normalized_clashes = normalizing_factor * clashes
        return normalized_clashes

    def rho_at_rank(self, rank, decay_function=None):
        return self.pivot_clashes_at_rank(rank=rank, decay_function=decay_function)

    @staticmethod
    def _count_pivots_in_fingered_stream(one_stream):
        big_n = len(one_stream)
        pivot_count = 0
        for i in range(big_n - 1): # the last note position cannot have a clash.
            pf_a, pf_b, direction = DEvaluation._get_basic_bigram_info(one_stream, i)
            if DEvaluation._is_pivot(pf_a, pf_b, direction):
                pivot_count += 1
        return pivot_count

    @staticmethod
    def abcdf_for_note_stream(stream):
        abcdf_str = ''
        for note in stream:
            pf = PianoFingering.fingering(note)
            abcdf_str += "{}{}".format(pf.strike_hand(), pf.strike_digit())
        return abcdf_str

    @staticmethod
    def _is_ascending(one_note, next_note):
        if one_note.pitch.midi < next_note.pitch.midi:
            return True
        return False

    @staticmethod
    def _is_descending(one_note, next_note):
        if one_note.pitch.midi > next_note.pitch.midi:
            return True
        return False

    @staticmethod
    def finger_contour_for_note_stream(stream):
        contour_string = ''
        prior_note = None
        for note in stream:
            if prior_note:
                if DEvaluation._is_ascending(one_note=prior_note, next_note=note):
                    contour_string += '/'
                elif DEvaluation._is_descending(one_note=prior_note, next_note=note):
                    contour_string += "\\"
                else:
                    contour_string += '-'
            pf = PianoFingering.fingering(note)
            contour_string += "{}{}".format(pf.strike_hand(), pf.strike_digit())
            prior_note = note
        return contour_string

    def print_pivot_count_report(self):
        human_stream = self._human_note_stream
        # human_abcdf = DEvaluation.abcdf_for_note_stream(human_stream)
        human_contour = DEvaluation.finger_contour_for_note_stream(human_stream)
        human_count = DEvaluation._count_pivots_in_fingered_stream(human_stream)
        print("PIVOT COUNTS")
        print("Human: {} Count: {}".format(human_contour, human_count))
        for i in range(len(self._system_note_streams)):
            system_stream = self._system_note_streams[i]
            # system_abcdf = DEvaluation.abcdf_for_note_stream(system_stream)
            system_contour = DEvaluation.finger_contour_for_note_stream(system_stream)
            system_count = DEvaluation._count_pivots_in_fingered_stream(system_stream)
            print("Sys {}: {} Count: {}".format(i + 1, system_contour, system_count))
        print("\n")

    def prob_stop_at_rank(self, rank, delta_function=None, rho_decay_function=None, big_delta_decay_function=None):
        big_delta_value = self.big_delta_at_rank(rank=rank, delta_function=delta_function,
                                                 decay_function=big_delta_decay_function)
        big_n = self._human_score.note_count(staff=self._staff)
        rho_value = self.rho_at_rank(rank=rank, decay_function=rho_decay_function)
        prob_at_rank = (1.0 - (big_delta_value / big_n)) / (math.pow(2, rho_value))
        return prob_at_rank