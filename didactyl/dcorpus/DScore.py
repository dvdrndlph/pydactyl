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
import re
from music21 import *
from didactyl.dactyler import Constant
from .DPart import DPart


class DScore:
    @staticmethod
    def _voice_id(abc_voice):
        reggie = r'^V:\s*(\d+)'
        matt = re.search(reggie, abc_voice.tokens[0].src)
        if matt:
            voice_id = matt.group(1)
            return int(voice_id)
        return None

    def __init__(self, music21_stream=None, segmenter=None, abc_handle=None, voice_map=None, abcd_header=None):
        self._abcd_header = abcd_header
        if music21_stream:
            self._combined_d_part = DPart(music21_stream=music21_stream, segmenter=segmenter)
            self._score = music21_stream
            meta = self._score[0]
            self._title = meta.title
        elif abc_handle:
            self._title = abc_handle.getTitle()
            music21_stream = abcFormat.translate.abcToStreamScore(abc_handle)
            self._combined_d_part = DPart(music21_stream=music21_stream, segmenter=segmenter)

            self._lower_d_part = None
            self._upper_d_part = None

            ah_array = abc_handle.splitByVoice()
            voices = []
            headers = None
            for ah in ah_array:
                if ah.hasNotes():
                    voices.append(ah)
                else:
                    if not headers:
                        headers = ah
                    else:
                        headers = headers + ah

            if len(voices) > 2 and not voice_map:
                raise Exception("Mapping of voices to staves is required.")
            if not voice_map:
                voice_map = {1: Constant.STAFF_UPPER, 2: Constant.STAFF_LOWER}
            if len(voices) >= 2:
                upper_ah = headers
                lower_ah = headers
                for voice in voices:
                    voice_id = DScore._voice_id(voice)
                    if voice_map[voice_id] == Constant.STAFF_UPPER:
                        upper_ah = upper_ah + voice
                    else:
                        lower_ah = lower_ah + voice

                upper_stream = abcFormat.translate.abcToStreamScore(upper_ah)
                self._upper_d_part = DPart(music21_stream=upper_stream, segmenter=segmenter)
                lower_stream = abcFormat.translate.abcToStreamScore(lower_ah)
                self._lower_d_part = DPart(music21_stream=lower_stream, segmenter=segmenter)

    def is_monophonic(self):
        return self._combined_d_part.is_monophonic()

    def d_part(self, staff):
        if staff == "upper":
            return self.upper_d_part()
        elif staff == "lower":
            return self.lower_d_part()
        elif staff == "both":
            return self.combined_d_part()

    def combined_d_part(self):
        return self._combined_d_part

    def upper_d_part(self):
        return self._upper_d_part

    def lower_d_part(self):
        return self._lower_d_part

    def stream(self, staff="both"):
        if staff == "upper":
            return self.upper_stream()
        elif staff == "lower":
            return self.lower_stream()
        return self._combined_d_part.stream()

    def pitch_range(self, staff="both"):
        if staff == "upper":
            return self._upper_d_part.pitch_range()
        if staff == "lower":
            return self._lower_d_part.pitch_range()
        return self._combined_d_part.pitch_range()

    def upper_stream(self):
        if self._upper_d_part:
            return self._upper_d_part.stream()
        return None

    def orderly_note_stream_segments(self, staff="both"):
        if staff == "upper":
            return self.upper_orderly_note_stream_segments()
        elif staff == "lower":
            return self.lower_orderly_note_stream_segments()
        return self._combined_d_part.orderly_note_stream_segments()

    def upper_orderly_note_stream_segments(self):
        if self._upper_d_part:
            return self._upper_d_part.orderly_note_stream_segments()
        return None

    def lower_orderly_note_stream_segments(self):
        if self._lower_d_part:
            return self._lower_d_part.orderly_note_stream_segments()
        return None

    def orderly_note_stream(self, staff="both"):
        if staff == "upper":
            return self.upper_orderly_note_stream()
        elif staff == "lower":
            return self.lower_orderly_note_stream()
        return self._combined_d_part.orderly_note_stream()

    def upper_orderly_note_stream(self):
        if self._upper_d_part:
            return self._upper_d_part.orderly_note_stream()
        return None

    def lower_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.stream()
        return None

    def lower_orderly_note_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.orderly_note_stream()
        return None

    def part_count(self):
        if self._upper_d_part and self._lower_d_part:
            return 2
        return 1

    def abcd_header(self):
        return self._abcd_header

    def is_annotated(self):
        if self._abcd_header:
            return True
        return False

    def note_count(self, staff="both"):
        upper = self.upper_d_part()
        lower = self.lower_d_part()
        combo = self.combined_d_part()
        note_count = 0
        if staff == 'upper' and upper:
            note_count = len(upper.orderly_note_stream())
        if staff == 'lower' and lower:
            note_count = len(lower.orderly_note_stream())
        if staff == 'both' and combo:
            note_count = len(combo.orderly_note_stream())
        return note_count

    def _is_fully_annotated(self, staff="both", indices=[]):
        if not self.is_annotated():
            return False
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")

        note_count = len(d_part.orderly_note_stream())
        annot_index = 0
        for annot in self._abcd_header.annotations():
            if indices and annot_index not in indices:
                continue
            sf_count = annot.score_fingering_count(staff=staff)
            if sf_count != note_count:
                return False
            for i in range(sf_count):
                strike_digit = annot.strike_digit_at_index(index=i, staff=staff)
                if not strike_digit or strike_digit == 'x':
                    return False
            annot_index += 1
        return True

    def is_fully_annotated(self, staff="both", indices=[]):
        """
        :return: True iff a strike digit is assigned to every note in the score.
        """
        if not self.is_annotated():
            return False
        upper = self.upper_d_part()
        lower = self.lower_d_part()
        combo = self.combined_d_part()

        if (upper and staff == "upper") or (lower and staff == "lower"):
            return self.staff_is_fully_annotated(staff=staff, indices=indices)

        if upper:
            is_upper_good = self._is_fully_annotated(staff="upper", indices=indices)
            is_lower_good = self._is_fully_annotated(staff="lower", indices=indices)
            is_good = is_upper_good and is_lower_good
            return is_good

        annot_index = 0
        note_count = len(combo.orderly_note_stream())
        for annot in self._abcd_header.annotations():
            if indices and annot_index not in indices:
                continue
            sf_count = annot.score_fingering_count(staff="both")
            if sf_count != note_count:
                return False
        return True

    def abcdf(self, index=0, identifier=None, staff="both"):
        if not self._abcd_header:
            return None

        if staff == "both":
            if identifier:
                return self._abcd_header.abcdf(identifier=identifier)
            else:
                return self._abcd_header.abcdf(index=index)
        elif staff == "upper":
            return self.upper_abcdf(index=index, identifier=identifier)
        elif staff == "lower":
            return self.lower_abcdf(index=index, identifier=identifier)

        return None

    def upper_abcdf(self, index=0, identifier=None):
        if self._abcd_header:
            if identifier:
                return self._abcd_header.upper_abcdf(identifier=identifier)
            else:
                return self._abcd_header.upper_abcdf(index=index)
        return None

    def lower_abcdf(self, index=0, identifier=None):
        if self._abcd_header:
            if identifier:
                return self._abcd_header.lower_abcdf(identifier=identifier)
            else:
                return self._abcd_header.lower_abcdf(index=index)
        return None

    def title(self):
        return self._title
