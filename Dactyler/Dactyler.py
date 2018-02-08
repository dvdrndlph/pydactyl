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
import os


class Note:
    def __init__(self, m21_note, prior_note=None):
        self._m21_note = m21_note
        self._prior_note = prior_note

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
        for n in m21_score[1].getElementsByClass(music21.note.Note):
            if not prior_note:
                new_note = Note(n)
            else:
                new_note = Note(n, prior_note=prior_note)

            notes.append(new_note)
            prior_note = new_note
        return notes

    def is_black(self):
        if not self._m21_note:
            return False
        return Note.note_class_is_black[self._m21_note.pitch.pitchClass]

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
        delta = self._m21_note.pitch.midi - self._prior_note.m21_note.pitch.midi
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
    def advise(self, offset=0, first_finger=None):
        return

    def load_corpus(self, d_corpus=None, path=None):
        if d_corpus:
            self._d_corpus = d_corpus
        elif path:
            self._d_corpus = DCorpus.DCorpus(path)
        else:
            raise Exception("No corpus specified for Dactyler.")

    def evaluate_hamming(self):
        print("Hamming")

    def evaluate_natural(self):
        print("Au natural")

    def evaluate_pivot(self):
        print("Pivot")

    def evaluate_pivot(self):
        print("Pivot")

    def evaluate_reentry(self):
        print("Re-enter the Twilight Zone")
