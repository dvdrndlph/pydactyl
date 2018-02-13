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

    def __init__(self, hands=Constant.HANDS_RIGHT, chords=False):
        self._d_corpus = None
        self._hands = hands
        self._chords = chords
        timestamp = datetime.now().isoformat()
        self._log_file_path = '/tmp/dactyler_' + self.__class__.__name__ + '_' + timestamp + '.log'
        self._log = open(self._log_file_path, 'a')

    def __del__(self):
        self._log.close()
        if Dactyler.DELETE_LOG:
            os.remove(self._log_file_path)

    def squawk(self, msg):
        self._log.write(str(msg) + "\n")
        if Dactyler.SQUAWK_OUT_LOUD:
            print(str(msg) + "\n")

    def squeak(self, msg):
        self._log.write(str(msg))
        if Dactyler.SQUAWK_OUT_LOUD:
            print(str(msg))

    @abstractmethod
    def advise(self, score_index=0, staff="upper", offset=0, first_finger=None):
        return

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
        if method == "pivot":
            cost = Constant.PIVOT_EDIT_DISTANCES[(one, other)]
            return cost

    def score_note_count(self, score_index=0, staff="both"):
        d_score = self._d_corpus.d_score_by_index(score_index)
        note_count = d_score.note_count(staff=staff)
        return note_count

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
        hamming_scores = []
        for gold_annot in hdr.annotations():
            hamming_score = 0
            for staff in staves:
                current_gold_hand = ">" if staff == "upper" else "<"
                current_test_hand = ">" if staff == "upper" else "<"

                test_sf_count = test_annot.score_fingering_count(staff=staff)
                gold_sf_count = gold_annot.score_fingering_count(staff=staff)

                if test_sf_count != gold_sf_count:
                    raise Exception("Length mismatch: {0} v. {1}".format(test_sf_count, gold_sf_count))
                for i in range(gold_sf_count):
                    gold_sf = gold_annot.score_fingering_at_index(index=i, staff=staff)
                    gold_strike = gold_sf.pf.fingering.strike
                    gold_hand = gold_strike.hand if gold_strike.hand else current_gold_hand
                    gold_digit = gold_strike.digit

                    test_sf = test_annot.score_fingering_at_index(index=i, staff=staff)
                    test_strike = test_sf.pf.fingering.strike
                    test_hand = test_strike.hand if test_strike.hand else current_test_hand
                    test_digit = test_strike.digit

                    current_gold_hand = gold_hand
                    current_test_hand = test_hand

                    hamming_score += Dactyler.strike_distance_cost(method=method,
                                                                   gold_hand=gold_hand,
                                                                   gold_digit=gold_digit,
                                                                   test_hand=test_hand,
                                                                   test_digit=test_digit)
            hamming_scores.append(hamming_score)

        return hamming_scores

    def evaluate_reentry(self, score_index=0, staff="upper"):
        print("Re-enter the Twilight Zone")
