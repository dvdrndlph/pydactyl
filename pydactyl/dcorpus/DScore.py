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
from music21 import abcFormat, stream 
from pydactyl.dactyler import Constant
from .DPart import DPart
from .PianoFingering import PianoFingering
from sklearn.metrics import cohen_kappa_score
# from krippendorff import alpha
from nltk.metrics.agreement import AnnotationTask
# from .ManualDSegmenter import ManualDSegmenter


class DScore:
    @staticmethod
    def _voice_id(abc_voice):
        reggie = r'^V:\s*(\d+)'
        matt = re.search(reggie, abc_voice.tokens[0].src)
        if matt:
            voice_id = matt.group(1)
            return int(voice_id)
        return None

    def abcd_header(self, abcd_header=None):
        if abcd_header:
            self._abcd_header = abcd_header
        return self._abcd_header

    def segmenter(self, segmenter=None):
        if segmenter:
            self._segmenter = segmenter
        return self._segmenter

    def finger(self, staff="both", d_annotation=None, id=1):
        if d_annotation:
            PianoFingering.finger_score(d_score=self, staff=staff,
                                        d_annotation=d_annotation, id=id)
        else:
            PianoFingering.finger_score(d_score=self, staff=staff, id=id)

    def interpolate(self, staff="both", d_annotation=None, id=1):
        self.finger(staff=staff, d_annotation=d_annotation, id=id)


    def __init__(self, music21_stream=None, segmenter=None, abc_handle=None,
                 voice_map=None, abcd_header=None):
        self._lower_d_part = None
        self._upper_d_part = None

        if music21_stream:
            self._combined_d_part = DPart(music21_stream=music21_stream, staff="both")
            self._score = music21_stream
            meta = self._score[0]
            self._title = meta.title
            parts = list(self._score.getElementsByClass(stream.Part))
            if len(parts) > 1:
                self._upper_d_part = DPart(music21_stream=parts[0], staff="upper")
                self._lower_d_part = DPart(music21_stream=parts[1], staff="lower")

        elif abc_handle:
            self._title = abc_handle.getTitle()
            music21_stream = abcFormat.translate.abcToStreamScore(abc_handle)
            self._combined_d_part = DPart(music21_stream=music21_stream, staff="both")

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
                self._upper_d_part = DPart(music21_stream=upper_stream, staff="upper")
                lower_stream = abcFormat.translate.abcToStreamScore(lower_ah)
                self._lower_d_part = DPart(music21_stream=lower_stream, staff="lower")

        self._abcd_header = abcd_header
        # self.abcd_header(abcd_header)
        self._segmenter = None
        self.segmenter(segmenter)

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

    def as_xml(self, staff="both"):
        from music21.musicxml import m21ToXml
        strm = self.stream(staff=staff)
        sx = m21ToXml.ScoreExporter(strm)
        mx_score = sx.parse()
        xml_str = mx_score.decode('utf-8')
        return xml_str

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

    def orderly_note_stream_segments(self, staff="both", offset=0):
        if not self._segmenter or not self._abcd_header:
            single_stream = self.orderly_note_stream(staff=staff, offset=offset)
            return [single_stream]

        self._segmenter.d_annotation(self._abcd_header.annotation())  # Phrases are marked in the first annotation.
        if staff == "upper":
            return self._segmenter.segment_to_orderly_streams(d_part=self._upper_d_part, offset=offset)
        elif staff == "lower":
            return self._segmenter.segment_to_orderly_streams(d_part=self._lower_d_part, offset=offset)
        return self._segmenter.segment_to_orderly_streams(d_part=self._combined_d_part, offset=offset)

    def orderly_note_stream(self, staff="both", offset=0):
        if staff == "upper":
            return self.upper_orderly_note_stream(offset=offset)
        elif staff == "lower":
            return self.lower_orderly_note_stream(offset=offset)
        return self._combined_d_part.orderly_note_stream(offset=offset)

    def upper_orderly_note_stream(self, offset=0):
        if self._upper_d_part:
            return self._upper_d_part.orderly_note_stream(offset=offset)
        else:
            return self._combined_d_part.orderly_note_stream(offset=offset)
        return None

    def lower_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.stream()
        return None

    def lower_orderly_note_stream(self, offset=0):
        if self._lower_d_part:
            return self._lower_d_part.orderly_note_stream(offset=offset)
        return None

    def part_count(self):
        if self._upper_d_part and self._lower_d_part:
            return 2
        return 1

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

    def assert_consistent_abcd(self, staff="both"):
        note_count = self.note_count(staff=staff)
        first_annot = self._abcd_header.annotation(index=0)
        sf_count = first_annot.score_fingering_count(staff=staff)
        if sf_count != note_count:
            raise Exception("Score fingering count does not match note count: {} != {}".format(sf_count, note_count))
        self._abcd_header.assert_consistent(staff=staff)

    def cohens_kappa(self, one_id, other_id, staff="both", common_id=None):
        one_annot = self._abcd_header.annotation_by_id(identifier=one_id)
        other_annot = self._abcd_header.annotation_by_id(identifier=other_id)
        common_annot = None
        if common_id:
            common_annot = self._abcd_header.annotation_by_id(identifier=common_id)

        one = []
        other = []
        common = []
        if staff == "upper" or staff == "both":
            one.extend(one_annot.handed_strike_digits(staff="upper"))
            other.extend(other_annot.handed_strike_digits(staff="upper"))
            if common_id:
                common.extend(common_annot.handed_strike_digits(staff="upper"))
        if staff == "lower" or staff == "both":
            one.extend(one_annot.handed_strike_digits(staff="lower"))
            other.extend(other_annot.handed_strike_digits(staff="lower"))
            if common_id:
                common.extend(common_annot.handed_strike_digits(staff="lower"))

        one_clean = one
        other_clean = other
        if common_id:
            one_clean = []
            other_clean = []
            for i in range(len(common)):
                if common[i] == 'x':
                    one_clean.append(one[i])
                    other_clean.append(other[i])

        labels = ['>1', '>2', '>3', '>4', '>5', '<1', '<2', '<3', '<4', '<5']
        pair_counts = {}
        all_labels = list(labels)
        all_labels.append('x')
        for label_1 in all_labels:
            for label_2 in all_labels:
                label_pair = "{}_{}".format(label_1, label_2)
                pair_counts[label_pair] = 0

        for i in range(len(one_clean)):
            pair_key = "{}_{}".format(one_clean[i], other_clean[i])
            pair_counts[pair_key] += 1
        kappa = cohen_kappa_score(one_clean, other_clean, labels=labels)
        return kappa, pair_counts

    # def krippendorffs_alpha(self, indices=[], segregate=False):
        # fingerings = list(upper_rh_advice)
        # fingerings.pop(0)
        # finger_ints = list(map(int, fingerings))
        # exercise_upper_gold.append(finger_ints)
        # krip = alpha(reliability_data=exercise_upper_gold, level_of_measurement='interval')
        # exercise_upper_gold.pop()
        # return krip

    def _is_fully_annotated(self, staff="both", indices=[]):
        if not self.is_annotated():
            return False
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")

        ons = d_part.orderly_note_stream()
        note_count = len(ons)
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
            return self._is_fully_annotated(staff=staff, indices=indices)

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
