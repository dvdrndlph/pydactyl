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
from pydactyl.dactyler import Constant
import music21
from .PianoFingering import PianoFingering
from .Ruler import ImaginaryBlackKeyRuler, NOTE_CLASS_IS_BLACK


class DNote:
    ASCENDING = 1
    DESCENDING = -1
    LATERAL = 0
    ASCENDING_CHAR = '/'
    DESCENDING_CHAR = "\\"
    LATERAL_CHAR = '-'
    DIRECTION_MAP = {
        ASCENDING_CHAR: ASCENDING,
        ASCENDING: ASCENDING_CHAR,
        DESCENDING_CHAR: DESCENDING,
        DESCENDING: DESCENDING_CHAR,
        LATERAL: LATERAL_CHAR,
        LATERAL_CHAR: LATERAL
    }

    BALLIAUW_RULER = ImaginaryBlackKeyRuler()

    def __init__(self, m21_note, prior_note=None):
        self._m21_note = m21_note
        self._prior_note = prior_note

    def _pitches_match(self, other):
        if self.midi() != other.midi():
            return False
        if self.prior_midi() != other.prior_midi():
            return False
        return True

    def piano_fingering(self):
        if PianoFingering.fingerings(self.m21_note()):
            pf = PianoFingering.fingering(self.m21_note())
            return pf
        return None

    def __eq__(self, other):
        if not isinstance(other, DNote):
            return False
        if not self._pitches_match(other):
            return False
        if self.piano_fingering() != other.piano_fingering():
            return False
        return True

    def __hash__(self):
        # return hash((self.midi(), self.prior_midi(), self.piano_fingering()))
        return hash(self.piano_fingering())

    def __repr__(self):
        my_str = "DNote({} Prior:{} {})".format(
            self.midi(), self.prior_midi(), self.piano_fingering())
        return my_str

    def m21_note(self):
        return self._m21_note

    def prior_note(self):
        return self._prior_note

    def prior_midi(self):
        if self._prior_note is None:
            return None
        return self._prior_note.midi()

    def velocity(self, value=None):
        """
        Return MIDI velocity of note or set it to value.
        """
        if value is not None:
            self._m21_note.volume.velocity = value
        return self._m21_note.volume.velocity

    def duration(self, value=None):
        """
        Return the quarterLength duration of the note or set it to value.
        """
        if value is not None:
            self._m21_note.duration = value
        return self._m21_note.duration.quarterLength

    note_class_is_cramped = {
        0: False,   # C
        1: False,   # C#
        2: True,    # D
        3: False,   # Eb
        4: False,   # E
        5: False,   # F
        6: False,   # F#
        7: True,    # G
        8: False,   # G#
        9: True,    # A
        10: False,  # Bb
        11: False   # B
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

    def is_cramped(self):
        if not self._m21_note:
            return False
        return DNote.note_class_is_cramped[self._m21_note.pitch.pitchClass]

    def is_black(self):
        if not self._m21_note:
            return False
        return NOTE_CLASS_IS_BLACK[self._m21_note.pitch.pitchClass]

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

    def is_descending(self):
        if not self.prior_midi() or not self.midi():
            return False
        if self.midi() < self.prior_midi():
            return True
        return False

    def is_repeating(self):
        if not self.prior_midi() or not self.midi():
            return False
        if self.midi() == self.prior_midi():
            return True
        return False

    def signed_semitone_delta(self):
        if self.prior_midi() is None:
            return None
        delta = self.midi() - self.prior_midi()
        return delta

    def signed_balliauw_delta(self):
        if self.prior_midi() is None:
            raise Exception("No prior midi for DNote.")
        distance = DNote.BALLIAUW_RULER.distance(from_midi=self.prior_midi(), to_midi=self.midi())
        return distance

    def signed_semitone_delta_from(self, earlier_note):
        delta = self.midi() - earlier_note.midi()
        return delta

    def signed_balliauw_delta_from(self, earlier_note):
        distance = DNote.BALLIAUW_RULER.distance(from_midi=earlier_note.midi(), to_midi=self.midi())
        return distance

    def semitone_delta(self):
        if self.prior_midi() is None:
            return None
        delta = self.midi() - self.prior_midi()
        delta = -1 * delta if delta < 0 else delta
        return delta

    def is_between(self, note_x, note_y):
        if not note_x or not note_y:
            return False

        midi = self.midi()
        midi_x = note_x.midi()
        midi_y = note_y.midi()

        if not midi or not midi_x or not midi_y:
            return False

        if midi_y > midi > midi_x:
            return True
        if midi_y < midi < midi_x:
            return True

        return False


class AnnotatedDNote(DNote):
    def __init__(self, m21_note, prior_note=None, strike_hand=None, strike_digit=None,
                 release_hand=None, release_digit=None):
        super().__init__(m21_note=m21_note, prior_note=prior_note)
        self._strike_hand = strike_hand
        self._strike_digit = strike_digit
        self._release_hand = release_hand
        self._release_digit = release_digit

    def __eq__(self, other):
        if not isinstance(other, AnnotatedDNote):
            return False
        return self._strike_hand == other._strike_hand and \
               self._strike_digit == other._strike_digit and \
               self._release_hand == other._release_hand and \
               self._release_digit == other._release_digit

    def __hash__(self):
        return hash((self._m21_note, self._prior_note,
                     self._strike_hand, self._strike_digit,
                     self._release_hand, self._release_digit))

    def strike_hand(self):
        return self._strike_hand

    def strike_digit(self):
        return self._strike_digit

    def release_hand(self):
        return self._release_hand

    def release_digit(self):
        return self._release_digit

    def tuple_str(self):
        return "%s:%s" % (self.midi(), self.strike_digit())

    def is_pivot(self):
        prior_ad_note = self.prior_note()
        if prior_ad_note is None:
            return False

        if prior_ad_note.strike_hand() != self.strike_hand():
            return False  # Pivots are one-handed maneuvers

        hand = prior_ad_note.strike_hand()

        if hand == ">":
            if self.strike_digit() == 1 and prior_ad_note.strike_digit() != 1 and \
                    not self.is_descending():
                return True
            if prior_ad_note.strike_digit() == 1 and self.strike_digit() != 1 and \
                    not self.is_ascending():
                return True
        else:
            if self.strike_digit() == 1 and prior_ad_note.strike_digit() != 1 and \
                    not self.is_ascending():
                return True
            if prior_ad_note.strike_digit() == 1 and self.strike_digit() != 1 and \
                    not self.is_descending():
                return True
        return False
