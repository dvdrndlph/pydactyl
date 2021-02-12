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


class DEvalFunction:
    @staticmethod
    def delta_hamming(one_note, other_note):
        one_pf = PianoFingering.fingering(one_note)
        other_pf = PianoFingering.fingering(other_note)
        if one_pf.strike_digit() != other_pf.strike_digit():
            return 1.0
        elif one_pf.strike_hand() != other_pf.strike_hand():
            return 1.0
        return 0.0

    @staticmethod
    def delta_long_short(one_note, other_note, epsilon=0.5):
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
            return 1.0 - epsilon
        return 1.0

    @staticmethod
    def interchangeable_distance(one_note, other_note, epsilon=1.0):
        """
        Return the edit distance between two fingered trigrams, assuming notes
        one and three are fingered the same, given the two fingered middle notes.
        :param one_note: A fingered middle note of a trigram.
        :param other_note: Another fingered middle note of the same sequence..
        :return: 0.0 if fingerings are the same, 1.0 - epsilon if fingerings are
                 arbitrary per theory notion that adjacent long fingers are
                 essentially interchangeable, 1.0 otherwise.
        """
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
            return 1.0 - epsilon
        return 1.0

    @staticmethod
    def _is_unigram_match(one_note_stream, other_note_stream, index):
        if len(one_note_stream) != len(other_note_stream):
            raise Exception("Mismatched note streams")
        if index < 0:
            return True
        if index >= len(one_note_stream):
            return True

        one_note = one_note_stream[index]
        other_note = other_note_stream[index]
        one_pf = PianoFingering.fingering(one_note)
        other_pf = PianoFingering.fingering(other_note)
        if one_pf.strike_hand() != other_pf.strike_hand():
            return False
        if one_pf.strike_digit() == other_pf.strike_digit():
            return True
        return False

    @staticmethod
    def tau(one_note_stream, other_note_stream, index, interchange_function=None, epsilon=1.0):
        note_count = len(one_note_stream)
        check_count = len(other_note_stream)
        if note_count != check_count:
            raise Exception("Note stream count mismatch")
        if index >= note_count + 2:
            raise Exception("Note stream index out of range")

        if not DEvalFunction._is_unigram_match(one_note_stream, other_note_stream, index):
            return 1.0
        if not DEvalFunction._is_unigram_match(one_note_stream, other_note_stream, index - 2):
            return 1.0
        if DEvalFunction._is_unigram_match(one_note_stream, other_note_stream, index - 1):
            return 0.0

        one_note = one_note_stream[index - 1]
        other_note = other_note_stream[index - 1]
        if interchange_function is not None:
            return interchange_function(one_note=one_note, other_note=other_note, epsilon=epsilon)
        return 1.0

    @staticmethod
    def tau_trigram(one_note_stream, other_note_stream, index, epsilon=0.0):
        return DEvalFunction.tau(one_note_stream=one_note_stream, other_note_stream=other_note_stream,
                                 index=index, interchange_function=None, epsilon=epsilon)

    @staticmethod
    def tau_nuanced(one_note_stream, other_note_stream, index, epsilon=1.0):
        interchange_function = DEvalFunction.interchangeable_distance
        return DEvalFunction.tau(one_note_stream=one_note_stream, other_note_stream=other_note_stream,
                                 index=index, interchange_function=interchange_function, epsilon=epsilon)

    @staticmethod
    def tau_relaxed(one_note_stream, other_note_stream, index, epsilon=1.0):
        note_count = len(one_note_stream)
        check_count = len(other_note_stream)
        if note_count != check_count:
            raise Exception("Note stream count mismatch")
        if index >= note_count + 2:
            raise Exception("Note stream index out of range")

        nuance_at_n = DEvalFunction.tau_nuanced(one_note_stream=one_note_stream,
                                                other_note_stream=other_note_stream,
                                                index=index, epsilon=epsilon)
        if nuance_at_n == 0:
            return 0.0
        if nuance_at_n == 1:
            return 1.0

        nuance_minus_1 = DEvalFunction.tau_nuanced(one_note_stream=one_note_stream,
                                                   other_note_stream=other_note_stream,
                                                   index=index-1, epsilon=epsilon)
        if nuance_minus_1 == 1:
            return 1.0

        nuance_minus_2 = DEvalFunction.tau_nuanced(one_note_stream=one_note_stream,
                                                   other_note_stream=other_note_stream,
                                                   index=index - 2, epsilon=epsilon)
        if nuance_minus_2 == 1:
            return 1.0

        return 1.0 - epsilon

    @staticmethod
    def decay_uniform(big_n, n):
        return big_n - n + 1

    @staticmethod
    def decay_none(big_n, n):
        return 1

    @staticmethod
    def rho_power2(rho_value):
        return math.pow(2, rho_value)

    @staticmethod
    def rho_plus1(rho_value):
        return rho_value + 1.0

    @staticmethod
    def phi_inverse(rank):
        return 1/rank


class DEvaluation:
    """
    Class to determine how well each of the top ranked outputs of a model
    system perform against a given gold-standard human.
    """
    def __init__(self, human_score=None, system_scores=[], staff="both",
                 delta_function=DEvalFunction.delta_hamming,
                 tau_function=DEvalFunction.tau_trigram,
                 decay_function=DEvalFunction.decay_none,
                 rho_function=None, rho_decay_function=DEvalFunction.decay_none,
                 epsilon=1.0, phi=DEvalFunction.phi_inverse, full_context=True):
        """
        Initialize a new DEvaluation object.
        :param human_score:
        :param system_scores:
        :param staff:
        :param delta_function:
        :param tau_function:
        :param decay_function:
        :param rho_function:
        :param rho_decay_function:
        :param epsilon:
        """
        self._staff = staff
        self._human_score = None
        self._human_note_stream = None
        self.human_score(d_score=human_score)
        self._system_scores = []
        self._system_note_streams = []
        self.system_scores(system_scores)
        self._delta_function = delta_function
        self._tau_function = tau_function
        self._decay_function = decay_function
        self._rho_function = rho_function
        self._rho_decay_function = rho_decay_function
        self._epsilon = epsilon
        self._phi = phi
        self._full_context = full_context

    def parameterize(self, delta_function=DEvalFunction.delta_hamming,
                     tau_function=DEvalFunction.tau_trigram,
                     decay_function=DEvalFunction.decay_none,
                     rho_function=None, rho_decay_function=DEvalFunction.decay_none,
                     epsilon=1.0, phi=DEvalFunction.phi_inverse, full_context=True):
        self._delta_function = delta_function
        self._tau_function = tau_function
        self._decay_function = decay_function
        self._rho_function = rho_function
        self._rho_decay_function = rho_decay_function
        self._epsilon = epsilon
        self._phi = phi
        self._full_context = full_context

    def delta_function(self, delta_function=None):
        self._delta_function = delta_function

    def tau_function(self, tau_function=None):
        self._tau_function = tau_function

    def decay_function(self, decay_function=None):
        self._decay_function = decay_function

    def rho_function(self, rho_function=None):
        self._rho_function = rho_function

    def rho_decay_function(self, rho_decay_function=None):
        self._rho_decay_function = rho_decay_function

    def epsilon(self, epsilon=None):
        if epsilon:
            self._epsilon = epsilon
        return self._epsilon

    def phi(self, phi=None):
        if phi:
            self._phi = phi
        return self._phi

    def full_context(self, full_context=None):
        if full_context:
            self._phi = full_context
        return self._full_context

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

    def trigram_big_delta_at_rank(self, rank, normalized=False):
        index = rank - 1
        human_stream = self._human_note_stream
        system_stream = self._system_note_streams[index]
        if len(human_stream) != len(system_stream):
            raise Exception("Mismatched orderly note streams")

        big_n = self._human_score.note_count(staff=self._staff)
        if self._full_context:
            # Evaluate each note in its full trigram context.
            # (Each note participates in three trigram comparisons.)
            big_n += 2

        # We need to normalize measure so that big_delta <= big_n.
        normalizing_factor = DEvaluation._normalizing_factor(big_n=big_n, decay_function=self._decay_function)
        distance = 0
        for i in range(big_n):
            decay_weight = 1
            if self._decay_function:
                decay_weight = self._decay_function(big_n=big_n, n=i+1)
            tau_value = self._tau_function(one_note_stream=human_stream, other_note_stream=system_stream,
                                           epsilon=self._epsilon, index=i)
            distance += decay_weight * tau_value
        big_delta = normalizing_factor * distance

        if normalized:
            # Caller wants value normalized between 0 and 1.
            big_delta /= big_n
        return big_delta

    def big_delta_at_rank(self, rank, normalized=False):
        index = rank - 1
        human_stream = self._human_note_stream
        system_stream = self._system_note_streams[index]
        if len(human_stream) != len(system_stream):
            raise Exception("Mismatched orderly note streams")

        big_n = self._human_score.note_count(staff=self._staff)
        normalizing_factor = DEvaluation._normalizing_factor(big_n=big_n, decay_function=self._decay_function)
        distance = 0
        for i in range(big_n):
            decay_weight = 1
            if self._decay_function:
                decay_weight = self._decay_function(big_n=big_n, n=i+1)
            delta_value = self._delta_function(one_note=human_stream[i], other_note=system_stream[i])
            distance += decay_weight * delta_value
        big_delta = normalizing_factor * distance

        if normalized:
            # Caller wants value normalized between 0 and 1.
            big_delta /= big_n
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

    def pivot_clashes_at_rank(self, rank):
        index = rank - 1
        one_stream = self._human_note_stream
        other_stream = self._system_note_streams[index]
        big_n = len(one_stream)
        normalizing_factor = DEvaluation._normalizing_factor(big_n=big_n, decay_function=self._decay_function)

        clashes = 0.0
        for i in range(big_n - 1): # the last note position cannot have a clash.
            a_pf, b_pf, direction = DEvaluation._get_basic_bigram_info(one_stream, i)
            x_pf, y_pf, direction = DEvaluation._get_basic_bigram_info(other_stream, i)
            is_clash = DEvaluation._is_pivot_clash(a_pf, b_pf, x_pf, y_pf, direction)
            if is_clash:
                if self._decay_function:
                    clashes += self._decay_function(big_n=big_n, n=i+1)
                else:
                    clashes += 1
        normalized_clashes = normalizing_factor * clashes
        return normalized_clashes

    def rho_at_rank(self, rank):
        return self.pivot_clashes_at_rank(rank=rank)

    @staticmethod
    def _count_pivots_in_fingered_stream(one_stream):
        big_n = len(one_stream)
        pivot_count = 0
        for i in range(big_n - 1):  # the last note position cannot have a clash.
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
            system_contour = DEvaluation.finger_contour_for_note_stream(system_stream)
            system_count = DEvaluation._count_pivots_in_fingered_stream(system_stream)
            print("Sys {}: {} Count: {}".format(i + 1, system_contour, system_count))
        print("\n")

    def prob_satisfied(self, rank):
        big_delta_value = self.big_delta_at_rank(rank=rank)
        big_n = self._human_score.note_count(staff=self._staff)
        rho_value = self.rho_at_rank(rank=rank)
        prob_at_rank = 1.0 - (big_delta_value / big_n)
        if self._rho_function:
            prob_at_rank /= self._rho_function(rho_value)
        return prob_at_rank

    def trigram_prob_satisfied(self, rank):
        big_delta_value = self.trigram_big_delta_at_rank(rank=rank)
        big_n = self._human_score.note_count(staff=self._staff)
        if self._full_context:
            big_n += 2
        rho_value = self.rho_at_rank(rank=rank)
        prob_at_rank = 1.0 - (big_delta_value / big_n)
        if self._rho_function:
            prob_at_rank /= self._rho_function(rho_value)
        return prob_at_rank

    def expected_reciprocal_rank(self, trigram=False):
        system_score_count = len(self._system_scores)
        prob_still_going = 1.0
        err = 0.0
        for r in range(1, system_score_count + 1):
            if trigram:
                prob_happy = self.trigram_prob_satisfied(rank=r)
            else:
                prob_happy = self.prob_satisfied(rank=r)
            discount_factor = self._phi(rank=r)
            err += (discount_factor * prob_still_going * prob_happy)
            prob_still_going *= (1 - prob_happy)
        return err
