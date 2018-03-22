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
import pickle
import re
from datetime import datetime
import music21
from Dactyler import Constant
from DCorpus import DCorpus
from DCorpus.DAnnotation import DAnnotation
import os


class DNote:
    def __init__(self, m21_note, prior_note=None):
        self._m21_note = m21_note
        self._prior_note = prior_note

    def m21_note(self):
        return self._m21_note

    def prior_note(self):
        return self._prior_note

    def __str__(self):
        my_str = "MIDI {0}".format(self.m21_note.midi)
        return my_str

    note_class_is_black = {
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

    @staticmethod
    def note_list(m21_score):
        prior_note = None
        notes = []
        for n in m21_score.getElementsByClass(music21.note.Note):
            if not prior_note:
                new_note = DNote(n)
            else:
                new_note = DNote(n, prior_note=prior_note)

            notes.append(new_note)
            prior_note = new_note
        return notes

    def is_black(self):
        if not self._m21_note:
            return False
        return DNote.note_class_is_black[self._m21_note.pitch.pitchClass]

    def is_white(self):
        if not self._m21_note:
            return False
        return not self.is_black()

    def color(self):
        if not self._m21_note:
            return None
        if self.is_black():
            return Constant.BLACK
        return Constant.WHITE

    def prior_color(self):
        if self._prior_note:
            return self._prior_note.color()
        return None

    def midi(self):
        if self._m21_note:
            return self._m21_note.pitch.midi
        return None

    def prior_midi(self):
        if self._prior_note:
            return self._prior_note.midi()
        return None

    def is_ascending(self):
        if not self.prior_midi() or not self.midi():
            return False
        if self.midi() > self.prior_midi():
            return True
        return False

    def semitone_delta(self):
        delta = self.midi() - self.prior_midi()
        delta = -1 * delta if delta < 0 else delta
        return delta


class Dactyler(ABC):
    """Base class for all Didactyl algorithms."""

    # FIXME: The log should be timestamped and for the specific algorithm being used.
    SQUAWK_OUT_LOUD = False
    DELETE_LOG = True

    def __init__(self):
        self._d_corpus = None
        timestamp = datetime.now().isoformat()
        self._log_file_path = '/tmp/dactyler_' + self.__class__.__name__ + '_' + timestamp + '.log'
        self._log = open(self._log_file_path, 'a')

    def __del__(self):
        self._log.close()
        if Dactyler.DELETE_LOG:
            os.remove(self._log_file_path)

    @staticmethod
    def combine_abcdf_segments(segments):
        abcdf = ''
        current_hand = None
        for seg in segments:
            for ch in seg:
                if ch == "<" or ch == ">":
                    if ch != current_hand:
                        abcdf += ch
                        current_hand = ch
                else:
                    abcdf += ch
        return abcdf

    @staticmethod
    def hand_digit(digit, staff):
        if digit is None:
            return None

        handed_re = re.compile('^[<>]\d$')
        if handed_re.match(str(digit)):
            return digit

        staff_prefix = ">"
        if staff == "lower":
            staff_prefix = "<"
        handed_digit = staff_prefix + str(digit)
        return handed_digit

    @staticmethod
    def digit_hand(handed_digit):
        handed_re = re.compile('^([<>]{1})\d$')
        mat = handed_re.match(str(handed_digit))
        hand = mat.group(1)
        if hand != "<" and hand != ">":
            raise Exception("Ill-formed handed digit: {0}".format(handed_digit))
        return hand

    def squawk(self, msg):
        self._log.write(str(msg) + "\n")
        if Dactyler.SQUAWK_OUT_LOUD:
            print(str(msg))

    def squeak(self, msg):
        self._log.write(str(msg))
        if Dactyler.SQUAWK_OUT_LOUD:
            print(str(msg), end="")

    @staticmethod
    def one_note_advise(d_note, staff="upper", first_digit=None, last_digit=None):
        if staff != "upper" and staff != "lower":
            raise Exception("One note advice not available for {0} staff.".format(staff))
        if first_digit and last_digit and first_digit != last_digit:
            raise Exception("Ambiguous digit constraint: {0} and {1}".format(first_digit, last_digit))

        if staff == "upper":
            advice = ">"
        else:
            advice = "<"

        digit = "1"
        if first_digit:
            digit = str(first_digit)
        elif last_digit:
            digit = last_digit
        elif d_note.is_black():
            digit = "2"
        advice += digit

        return advice

    @abstractmethod
    def segment_advise(self, segment, staff, offset, handed_first_digit, handed_last_digit, top=None):
        pass

    def advise(self, score_index=0, staff="upper", offset=0, first_digit=None, last_digit=None, top=None):
        d_scores = self._d_corpus.d_score_list()
        if score_index >= len(d_scores):
            raise Exception("Score index out of range")

        d_score = d_scores[score_index]
        if staff == "both":
            if offset or first_digit or last_digit:
                raise Exception("Ambiguous use to offset and/or first/last digit for both staves.")
            upper_advice = self.advise(score_index=score_index, staff="upper")
            abcdf = upper_advice + "@"
            if d_score.part_count() > 1:
                lower_advice = self.advise(score_index=score_index, staff="lower")
                abcdf += lower_advice
            return abcdf

        if staff != "upper" and staff != "lower":
            raise Exception("Segregated advice is only dispensed one staff at a time.")

        handed_first_digit = Dactyler.hand_digit(digit=first_digit, staff=staff)
        handed_last_digit = Dactyler.hand_digit(digit=last_digit, staff=staff)

        if d_score.part_count() == 1:
            d_part = d_score.combined_d_part()
        else:
            # We support (segregated) left hand fingerings. By segregated, we
            # mean the right hand is dedicated to the upper staff, and the
            # left hand is dedicated to the lower staff.
                d_part = d_score.d_part(staff=staff)

        segments = d_part.orderly_note_stream_segments(offset=offset)
        segment_index = 0
        last_segment_index = len(segments) - 1
        advice_segments = []
        for segment in segments:
            segment_offset = 0
            segment_handed_first = None
            segment_handed_last = None
            if segment_index == 0:
                segment_offset = offset
                segment_handed_first = handed_first_digit
            if segment_index == last_segment_index:
                segment_handed_last = handed_last_digit

            segment_advice = self.segment_advise(segment=segment, staff=staff,
                                                 offset=segment_offset,
                                                 handed_first_digit=segment_handed_first,
                                                 handed_last_digit=segment_handed_last, top=top)
            advice_segments.append(segment_advice)
            segment_index += 1

        abcdf = Dactyler.combine_abcdf_segments(advice_segments)
        return abcdf

    def load_corpus(self, d_corpus=None, path=None):
        if d_corpus:
            self._d_corpus = d_corpus
        elif path:
            self._d_corpus = DCorpus.DCorpus(path)
        else:
            raise Exception("No corpus specified for Dactyler.")

    @staticmethod
    def strike_distance_cost(gold_hand, gold_digit, test_hand, test_digit, method="hamming"):
        if method == "hamming":
            if test_digit != gold_digit or test_hand != gold_hand:
                return 1
            else:
                return 0

        one = str(gold_hand) + str(gold_digit)
        other = str(test_hand) + str(test_digit)
        if method == "natural":
            cost = Constant.NATURAL_EDIT_DISTANCES[(one, other)]
            return cost
        elif method == "pivot":
            cost = Constant.PIVOT_EDIT_DISTANCES[(one, other)]
            return cost
        else:
            raise Exception("Unsupported method: {0}".format(method))

    def score_note_count(self, score_index=0, staff="both"):
        d_score = self._d_corpus.d_score_by_index(score_index)
        note_count = d_score.note_count(staff=staff)
        return note_count

    @staticmethod
    def _distance_and_loc(method, staff, test_annot, gold_annot, gold_offset=0, zero_cost=False):
        current_gold_hand = ">" if staff == "upper" else "<"
        current_test_hand = ">" if staff == "upper" else "<"

        test_sf_count = test_annot.score_fingering_count(staff=staff)
        gold_sf_count = gold_annot.score_fingering_count(staff=staff)

        adjusted_gold_sf_count = gold_sf_count - gold_offset
        if test_sf_count != adjusted_gold_sf_count:
            raise Exception("Length mismatch: test: {0} gold: {1}".format(test_sf_count, adjusted_gold_sf_count))

        score = 0
        i = None
        gold_digit = None
        for i in range(test_sf_count):
            gold_i = i + gold_offset
            gold_sf = gold_annot.score_fingering_at_index(index=gold_i, staff=staff)
            gold_strike = gold_sf.pf.fingering.strike
            gold_hand = gold_strike.hand if gold_strike.hand else current_gold_hand
            gold_digit = int(gold_strike.digit)

            test_sf = test_annot.score_fingering_at_index(index=i, staff=staff)
            test_strike = test_sf.pf.fingering.strike
            test_hand = test_strike.hand if test_strike.hand else current_test_hand
            test_digit = int(test_strike.digit)

            current_gold_hand = gold_hand
            current_test_hand = test_hand

            cost = Dactyler.strike_distance_cost(method=method,
                                                 gold_hand=gold_hand,
                                                 gold_digit=gold_digit,
                                                 test_hand=test_hand,
                                                 test_digit=test_digit)
            if zero_cost and cost:
                return cost, gold_i, gold_digit
            score += cost

        return score, i, gold_digit

    def _eval_strike_distance(self, method, staff, test_annot, gold_annot):
        (cost, location, gold_digit) = Dactyler._distance_and_loc(method=method, staff=staff,
                                                                  test_annot=test_annot, gold_annot=gold_annot)
        return cost

    def evaluate_strike_distance(self, method="hamming", score_index=0, staff="upper"):
        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        if staff == "both":
            staves = ['upper', 'lower']
        else:
            staves = [staff]

        test_abcdf = self.advise(score_index=score_index, staff=staff)
        test_annot = DAnnotation(abcdf=test_abcdf)
        hdr = d_score.abcd_header()
        scores = []
        for gold_annot in hdr.annotations():
            score = 0
            for staff in staves:
                score += self._eval_strike_distance(method=method, staff=staff,
                                                    test_annot=test_annot, gold_annot=gold_annot)
            scores.append(score)

        return scores

    def evaluate_strike_reentry(self, method="hamming", score_index=0, staff="upper", gold_indices=[]):
        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        if staff == "both":
            staves = ['upper', 'lower']
        else:
            staves = [staff]

        hdr = d_score.abcd_header()
        scores = {"upper": [], "lower": []}
        for staff in staves:
            current_gold_index = 0
            for gold_annot in hdr.annotations():
                score = 0
                if len(gold_indices) > 0 and current_gold_index not in gold_indices:
                    current_gold_index += 1
                    continue
                current_gold_index += 1

                test_abcdf = self.advise(score_index=score_index, staff=staff)
                self.squawk("COMPLETE ADVICE: {0}".format(test_abcdf))
                if staff == 'upper':
                    test_abcdf += '@'
                else:
                    test_abcdf = '@' + str(test_abcdf)
                test_annot = DAnnotation(abcdf=test_abcdf)
                (cost, loc, gold_digit) = self._distance_and_loc(zero_cost=True, method=method, staff=staff,
                                                                 test_annot=test_annot, gold_annot=gold_annot)
                score += cost
                self.squawk("GOLD: {0} staff for {1}".format(staff, gold_annot.abcdf(staff=staff, flat=True)))
                self.squawk("TEST: {0} staff for {1}".format(staff, test_abcdf))
                self.squawk("SCORE: {0} COST: {1} LOC: {2}".format(score, cost, loc))

                while cost > 0:
                    test_abcdf = self.advise(score_index=score_index, staff=staff,
                                             offset=loc, first_digit=gold_digit)
                    self.squawk("GOLD: {0} staff for {1}".format(staff, gold_annot.abcdf(staff=staff, flat=True)))
                    self.squawk("TRUNCATED ADVICE: {0}".format(test_abcdf))
                    if staff == 'upper':
                        test_abcdf += '@'
                    else:
                        test_abcdf = '@' + str(test_abcdf)
                    test_annot = DAnnotation(abcdf=test_abcdf)
                    (cost, loc, gold_digit) = self._distance_and_loc(zero_cost=True, gold_offset=loc,
                                                                     method=method, staff=staff,
                                                                     test_annot=test_annot, gold_annot=gold_annot)
                    score += cost
                    self.squawk("     score: {0} cost: {1} loc: {2}".format(score, cost, loc))

                scores[staff].append(score)

        total_scores = []
        if len(scores['upper']) > 0:
            for i in range(len(scores['upper'])):
                total_scores.append(scores['upper'][i])
                if len(scores['lower']) > 0:
                    total_scores[i] += scores['lower'][i]
        elif len(scores['lower']) > 0:
            for i in range(len(scores['lower'])):
                total_scores.append(scores['lower'][i])
        else:
            raise Exception("No scores found.")

        return total_scores

    def evaluate_pivot_alignment(self, score_index=0, staff="upper"):
        d_score = self._d_corpus.d_score_by_index(score_index)
        if not d_score.is_fully_annotated():
            raise Exception("Only fully annotated scores can be evaluated.")

        if staff == "both":
            staves = ['upper', 'lower']
        else:
            staves = [staff]

        test_abcdf = self.advise(score_index=score_index, staff=staff)
        test_annot = DAnnotation(abcdf=test_abcdf)
        hdr = d_score.abcd_header()
        scores = []
        for gold_annot in hdr.annotations():
            score = 0
            for staff in staves:
                score += self._eval_strike_distance(method=method, staff=staff,
                                                    test_annot=test_annot, gold_annot=gold_annot)
            scores.append(score)

        return scores
class TrainedDactyler(Dactyler):
    def __init__(self):
        super().__init__()
        self._smoother = None
        self._training = {}

    @abstractmethod
    def segment_advise(self, segment, staff, offset, handed_first_digit, handed_last_digit, top=None):
        pass

    @abstractmethod
    def train(self, d_corpus, staff="both", segregate=True, segmenter=None, annotation_indices=[]):
        return

    def retain(self, pickle_path=None, to_db=False):
        if pickle_path:
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(self._training, pickle_file, pickle.HIGHEST_PROTOCOL)
        elif to_db:
            raise Exception("Retaining pickled file to database not yet supported.")
        else:
            raise Exception("No retention destination specified.")

    def recall(self, pickle_path=None, pickle_db_id=None):
        if pickle_path:
            with open(pickle_path, 'rb') as pickle_file:
                self._training = pickle.load(pickle_file)
        elif pickle_db_id:
            raise Exception("Recalling pickled training from database not yet supported.")
        else:
            raise Exception("No source specified from which to recall training.")

    def training(self):
        return self._training

    def demonstrate(self):
        for k in self._training:
            print(k, ': ', self._training[k])

    def smoother(self, method=None):
        if method is None:
            return self._smoother
        else:
            self._smoother = method

    @abstractmethod
    def smooth(self):
        return

