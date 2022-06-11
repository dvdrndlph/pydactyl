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
import numpy as np
from music21 import abcFormat, stream, midi
from pydactyl.dactyler import Constant
from nltk.metrics.distance import binary_distance
from .DNote import DNote
from .DPart import DPart
from .ABCDHeader import ABCDHeader
from .PianoFingering import PianoFingering
from sklearn.metrics import cohen_kappa_score
from krippendorff import alpha
from nltk.metrics.agreement import AnnotationTask

# The locations of the various data in a bigram string
FROM_HAND_INDEX = 0
FROM_DIGIT_INDEX = 1
DIRECTION_INDEX = 2
TO_HAND_INDEX = 3
TO_DIGIT_INDEX = 4


class DScore:
    _bigram_t = 1
    _bigram_p = 1
    _bigram_big_n_bar = 16

    @staticmethod
    def file_to_string(file_path):
        string = ''
        file = open(file_path, "r")
        for line in file:
            string += line
        file.close()
        return string

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
            self._combined_d_part.segmenter(segmenter=segmenter)
            self._upper_d_part.segmenter(segmenter=segmenter)
            self._lower_d_part.segmenter(segmenter=segmenter)
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
                 midi_file_path=None, abcd_header_path=None,
                 voice_map=None, abcd_header=None, abc_body='', title=None):
        self._lower_d_part = None
        self._upper_d_part = None
        self._abc_body = abc_body
        self._title = title

        if midi_file_path is not None:
            music21_stream = DScore.score_via_midi(corpus_path=midi_file_path)

        if abcd_header_path is not None:
            abcd_str = DScore.file_to_string(file_path=abcd_header_path)
            abcd_header = ABCDHeader(abcd_str=abcd_str)

        if music21_stream:
            self._combined_d_part = DPart(music21_stream=music21_stream, staff="both")
            self._score = music21_stream
            meta = self._score[0]
            if self._title is None:
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
        self._segmenter = None
        self.segmenter(segmenter)

    def __str__(self):
        abcd_str = self._abcd_header.__str__()
        abcd_str += self._abc_body
        return abcd_str

    @staticmethod
    def bigram_t(t=None):
        if DScore._bigram_t is not None:
            DScore._bigram_t = t
        return DScore._bigram_t

    @staticmethod
    def bigram_p(p=None):
        if DScore._bigram_p is not None:
            DScore._bigram_p = p
        return DScore._bigram_p

    @staticmethod
    def bigram_big_n_bar(big_n_bar=None):
        if DScore._bigram_big_n_bar is not None:
            DScore._bigram_big_n_bar = big_n_bar
        return DScore._bigram_big_n_bar

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

    def ordered_offset_notes(self, staff="both", offset=0):
        if staff == "upper":
            return self.upper_ordered_offset_notes(offset=offset)
        elif staff == "lower":
            return self.lower_ordered_offset_notes(offset=offset)
        return self._combined_d_part.ordered_offset_notes(offset=offset)

    def upper_orderly_note_stream(self, offset=0):
        if self._upper_d_part:
            return self._upper_d_part.orderly_note_stream(offset=offset)
        else:
            return self._combined_d_part.orderly_note_stream(offset=offset)

    def upper_ordered_offset_notes(self, offset=0):
        if self._upper_d_part:
            return self._upper_d_part.ordered_offset_notes(offset=offset)
        else:
            return self._combined_d_part.ordered_offset_notes(offset=offset)

    def lower_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.stream()
        return None

    def lower_orderly_note_stream(self, offset=0):
        if self._lower_d_part:
            return self._lower_d_part.orderly_note_stream(offset=offset)
        return None

    def lower_ordered_offset_notes(self, offset=0):
        if self._lower_d_part:
            return self._lower_d_part.ordered_offset_notes(offset=offset)
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

    @staticmethod
    def unigram_labels(staff="both", wildcard=False, segregated=False):
        labels = ['>1', '>2', '>3', '>4', '>5', '<1', '<2', '<3', '<4', '<5']
        if segregated:
            if staff == "both":
                raise Exception("Cannot calculate segregated kappa for both staffs.")
            elif staff == "upper":
                labels = ['>1', '>2', '>3', '>4', '>5']
            else:
                labels = ['<1', '<2', '<3', '<4', '<5']
        if wildcard:
            labels.append('x')
        return labels

    def cohens_kappa_data(self, one_id, other_id, staff="both", common_id=None, wildcard=True, segregated=False):
        labels = DScore.unigram_labels(staff=staff, wildcard=wildcard, segregated=segregated)

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

        pair_counts = {}
        all_labels = list(labels)
        for label_1 in all_labels:
            for label_2 in all_labels:
                label_pair = "{}_{}".format(label_1, label_2)
                pair_counts[label_pair] = 0

        for i in range(len(one_clean)):
            pair_key = "{}_{}".format(one_clean[i], other_clean[i])
            if pair_key not in pair_counts:
                raise Exception("Bad key: {}".format(pair_key))
            pair_counts[pair_key] += 1
        data = {
            'pair_counts': pair_counts,
            'labels': labels,
            'one': one_clean,
            'other': other_clean
        }
        return data

    def cohens_kappa(self, one_id, other_id, staff="both", common_id=None, wildcard=True, segregated=False):
        data = self.cohens_kappa_data(one_id=one_id, other_id=other_id, staff=staff, common_id=common_id,
                                      wildcard=wildcard, segregated=segregated)
        kappa = cohen_kappa_score(data['one'], data['other'], labels=data['labels'])
        return kappa, data['pair_counts']

    def _note_indices_to_ignore(self, staff="both", common_id=None):
        ignore = {}
        if common_id:
            common_annot = self._abcd_header.annotation_by_id(identifier=common_id)
            note_index = 0
            if staff == "upper" or staff == "both":
                for hsd in common_annot.handed_strike_digits(staff="upper"):
                    if hsd is not None and hsd != 'x':
                        ignore[note_index] = True
                    note_index += 1
            if staff == "lower" or staff == "both":
                for hsd in common_annot.handed_strike_digits(staff="lower"):
                    if hsd is not None and hsd != 'x':
                        ignore[note_index] = True
                    note_index += 1
        return ignore

    @staticmethod
    def _is_good_bigram_sr_action(action):
        if action is None:
            return False
        if action[0] not in ('>', '<'):
            return False
        if action[1] not in (1, 2, 3, 4, 5):
            return False
        return True

    @staticmethod
    def _bigram_is_underspecified(sr_data, note_index):
        if sr_data is None:
            return True
        elif note_index != 0:
            prior_srd = sr_data[note_index - 1]
            if not DScore._is_good_bigram_sr_action(prior_srd['release']):
                return True
        else:
            if not DScore._is_good_bigram_sr_action(sr_data['strike']):
                return True
        return False

    def _bigram_note_indices_to_ignore(self, staff="upper", common_id=None):
        if staff not in ('upper', 'lower'):
            raise Exception("Bigram metrics work on one staff at a time.")

        ignore = {}
        if common_id:
            common_annot = self._abcd_header.annotation_by_id(identifier=common_id)
            note_index = 0
            for srd in common_annot.handed_strike_digits(staff=staff):
                if DScore._bigram_is_underspecified(sr_data=srd, note_index=note_index):
                    ignore[note_index] = True
                note_index += 1
        return ignore

    def trigram_strike_annotation_data(self, staff="upper"):
        """
        Data for feeding string tags to the NLTK AnnotationTask.
        Here we are generating a set of trigram labels that consider
        only strike hands and fingers, which is all we considered in the
        SMC2021 paper.
        :param staff:
        :param common_id:
        :return:
        """
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")
        data = {}
        for annot in self._abcd_header.annotations():
            strikes = []
            coder_id = annot.abcdf_id()
            strike_str_0 = 'xx'
            strike_str_1 = 'xx'
            for hsd in annot.handed_strike_digits(staff=staff):
                if len(hsd) != 2:
                    raise Exception("Unsupported fingering sequence")
                strike_str = hsd
                trigram_str = strike_str_0 + strike_str_1 + strike_str
                strikes.append(trigram_str)
                strike_str_0 = strike_str_1
                strike_str_1 = strike_str
            trigram_str = strike_str_0 + strike_str_1 + 'xx'
            strikes.append(trigram_str)
            trigram_str = strike_str_1 + 'xxxx'
            strikes.append(trigram_str)
            data[coder_id] = strikes
        return data

    def unigram_strike_annotation_data(self, staff="upper"):
        """
        Data for feeding string tags to the NLTK AnnotationTask.
        Here we are generating a set of unigram labels that consider
        only strike hands and fingers, which is all we considered in the
        SMC2021 paper.
        :param staff:
        :return: A dictionary (keyed by abcDF annotator ID) of lists of fingering strings.
        """
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")
        data = {}
        for annot in self._abcd_header.annotations():
            strikes = []
            coder_id = annot.abcdf_id()
            for hsd in annot.handed_strike_digits(staff=staff):
                if len(hsd) != 2:
                    raise Exception("Unsupported fingering sequence")
                strikes.append(hsd)
            data[coder_id] = strikes
        return data

    def orderly_d_notes(self, staff="upper", offset=0):
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")
        ordered_d_notes = d_part.orderly_d_notes(offset=offset)
        return ordered_d_notes

    def orderly_d_note_segments(self, staff="upper", offset=0):
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")
        orderly_segments = d_part.orderly_d_note_segments(offset=offset)
        return orderly_segments

    @staticmethod
    def score_via_midi(corpus_path):
        mf = midi.MidiFile()
        mf.open(corpus_path, attrib='rb')
        mf.read()
        mf.close()
        if len(mf.tracks) != 2:
            raise Exception("Two-track MIDI file required.")
        score = midi.translate.midiFileToStream(mf=mf, quantizePost=False)
        return score

    def _bigram_annotation_data(self, ids=None, staff="upper", common_id=None, offset=0):
        """
        Data for feeding bigram tags to the NLTK AnnotationTask.
        :param ids:
        :param staff:
        :param common_id:
        :return:
        """
        if ids is None:
            ids = []
        if staff == "upper":
            d_part = self.upper_d_part()
        elif staff == "lower":
            d_part = self.lower_d_part()
        else:
            raise Exception("Specific staff must be specified.")

        ordered_notes = d_part.orderly_d_notes(offset=offset)
        ignore = self._bigram_note_indices_to_ignore(staff=staff, common_id=common_id)
        data = []
        for coder_id in ids:
            note_index = 0
            annot = self._abcd_header.annotation_by_id(identifier=coder_id)
            for sr_data in annot.strike_release_data(staff=staff):
                if note_index in ignore or DScore._bigram_is_underspecified(sr_data=sr_data, note_index=note_index):
                    pass
                else:
                    d_note = ordered_notes[note_index]
                    direction = 0
                    prior_sr = None
                    if note_index != 0:
                        prior_sr = sr_data[note_index - 1]
                        if d_note.is_ascending():
                            direction = 1
                        elif d_note.is_descending():
                            direction = -1
                    sr = sr_data[note_index]
                    record = {'coder_id': coder_id,
                              'note_index': note_index,
                              'direction': direction,
                              'prior_sr': prior_sr,
                              'sr': sr}
                    data.append(record)
                note_index += 1
        return data

    @staticmethod
    def _all_same_hand(one, other):
        if one[FROM_HAND_INDEX] == one[TO_HAND_INDEX] and \
                one[TO_HAND_INDEX] == other[FROM_HAND_INDEX] and \
                other[FROM_HAND_INDEX] == other[TO_HAND_INDEX]:
            return True
        return False

    @staticmethod
    def _ascending(bigram_str):
        direction = bigram_str[DIRECTION_INDEX]
        if direction == DNote.ASCENDING_CHAR:
            return True
        return False

    @staticmethod
    def _descending(bigram_str):
        direction = bigram_str[DIRECTION_INDEX]
        if direction == DNote.DESCENDING_CHAR:
            return True
        return False

    @staticmethod
    def _pivot_clash(one, other):
        """
        Assumes all hands are the same.
        >>> uno = ">1"
        """
        hand = one[FROM_HAND_INDEX]
        direction = one[DIRECTION_INDEX]
        if direction == DNote.LATERAL_CHAR:
            return False

        a = int(one[FROM_HAND_INDEX])
        b = int(one[TO_HAND_INDEX])
        x = int(other[FROM_HAND_INDEX])
        y = int(other[TO_HAND_INDEX])
        if hand == '>':  # pivotclash_RH
            if (DScore._ascending(one) and 1 in (b, y) and b != y) or \
                    (DScore._descending(one) and 1 in (a, x) and a != x):
                return True
        elif hand == '<':  # pivotclash_LH
            if (DScore._descending(one) and 1 in (b, y) and b != y) or \
                    (DScore._ascending(one) and 1 in (a, x) and a != x):
                return True
        return False

    @staticmethod
    def _wpclashes(seq_x, seq_y, rotation=0):
        if rotation != 0:
            raise Exception("Rotation in radians (to temper or increase) equity effects not supported yet.")
            # We use Integral_0^4 -1x/2 + 2
            # Could adjust to Integral_0^4 -1x/4 + 3/2, but I am tired of knobs to support my
            # dime-store cognitive modeling of "equity." Alex could help here.

        big_n = len(seq_y)
        if len(seq_x) != big_n:
            raise Exception("Weighted pivot clashes not supported for unbalanced bigram label sequences.")
        wpclashes = 0
        for n in range(big_n):
            if DScore._pivot_clash(seq_x[n], seq_y[n]):
                clash_weight = big_n - n + 1
                wpclashes += clash_weight
        wpclashes *= (big_n / (2 * (big_n + 1)))  # Normalize between 0 and N
        return wpclashes

    @staticmethod
    def _pivot_clash_discount(sys_seq, human_seq, base=16):
        wpclashes = DScore._wpclashes(seq_x=sys_seq, seq_y=human_seq)
        val = 1.0 / (base**wpclashes)
        return val

    @staticmethod
    def _aligned(seq_x, seq_y):
        if len(seq_x) != len(seq_y):
            raise Exception("Alignment check not supported for unbalanced bigram label sequences.")
        for i in range(len(seq_x)):
            if DScore._pivot_clash(seq_x[i], seq_y[i]):
                return False
        return True

    @staticmethod
    def bigram_label_distance(one, other):
        t = DScore.bigram_t()
        p = DScore.bigram_p()
        n_bar = DScore.bigram_big_n_bar()

        # hfdhf or ^=hf
        if len(one) != len(other):
            raise Exception("Bigram labels to compare must be the same length.")
        if one[DIRECTION_INDEX] != other[DIRECTION_INDEX]:
            raise Exception("Mismatched pitch direction in bigrams being compared.")
        if not DScore._all_same_hand(one, other):
            return 1.0/p

        b = int(one[TO_DIGIT_INDEX])
        y = int(other[TO_DIGIT_INDEX])
        if one[FROM_HAND_INDEX] == '^':
            distance = abs(b - y)/(t * n_bar)
            return distance

        if DScore._pivot_clash(one, other):
            return 1.0/p

        a = int(one[FROM_DIGIT_INDEX])
        x = int(other[FROM_DIGIT_INDEX])
        distance = (abs(a - x) + abs(b - y)) / (t * n_bar)
        return distance

    @staticmethod
    def _bigram_datum_to_label(datum, is_first=False):
        direction_int = datum['direction']
        direction_str = DNote.DIRECTION_MAP[direction_int]
        if is_first:
            prior_hand = '^'
            prior_digit = '^'
        else:
            prior_hand = datum['prior_sr']['release'][0]
            prior_digit = datum['prior_sr']['release'][1]
        hand = datum['sr']['strike'][0]
        digit = datum['sr']['strike'][1]
        label = "{}{}{}{}{}".format(prior_hand, prior_digit, direction_str, hand, digit)
        record = [datum['coder_id'], datum['note_index'], label]
        return record

    def _nltk_bigram_staff_annotation_data(self, ids=None, staff="upper", common_id=None, offset=0):
        """
        Data for feeding the NLTK AnnotationTask.
        :param ids:
        :param staff:
        :param common_id:
        :return:
        """
        if ids is None:
            ids = []
        if staff not in ('upper', 'lower'):
            raise Exception("Annotation data collected one staff at a time.")

        nltk_data = []
        data = self._bigram_annotation_data(ids=ids, staff=staff, common_id=common_id, offset=offset)
        is_first = True
        for datum in data:
            if is_first:
                label = DScore._bigram_datum_to_label(datum=datum, is_first=is_first)
                is_first = False
            else:
                label = DScore._bigram_datum_to_label(datum=datum)
            record = [datum['coder_id'], datum['note_index'], label]
            nltk_data.append(record)
        return nltk_data

    def _pypi_bigram_staff_annotation_data(self, ids=None, staff="upper", common_id=None, offset=0):
        """
        Data for feeding the PyPI krippendorff module.
        :param ids:
        :param staff:
        :param common_id:
        :return:
        """
        if ids is None:
            ids = []
        if staff not in ('upper', 'lower'):
            raise Exception("Annotation data collected one staff at a time.")

        pypi_data = []
        data = self._bigram_annotation_data(ids=ids, staff=staff, common_id=common_id, offset=offset)
        is_first = True
        human_id = None
        human_data = {}
        for datum in data:
            if is_first:
                label = DScore._bigram_datum_to_label(datum=datum, is_first=is_first)
                is_first = False
            else:
                label = DScore._bigram_datum_to_label(datum=datum)
            if datum['coder_id'] != human_id:
                human_data = {}
                human_id = datum['coder_id']
                pypi_data.append(human_data)
            human_data[datum['note_index']] = label
        pypi_data.append(human_data)
        return pypi_data

    def _nltk_staff_annotation_data(self, ids=None, staff="upper", measurement='nominal', common_id=None):
        """
        The data suitable for feeding the NLTK AnnotationTask.
        :param ids:
        :param staff:
        :param common_id:
        :return:
        """
        if ids is None:
            ids = []
        if staff not in ('upper', 'lower'):
            raise Exception("Annotation data collected one staff at a time.")

        ignore = self._note_indices_to_ignore(staff=staff, common_id=common_id)
        data = []
        for coder_id in ids:
            note_index = 0
            annot = self._abcd_header.annotation_by_id(identifier=coder_id)
            for hsd in annot.handed_strike_digits(staff=staff):
                if hsd is not None and hsd != 'x' and note_index not in ignore:
                    record = [coder_id, note_index, hsd]
                    data.append(record)
                note_index += 1
        return data

    def _pypi_staff_annotation_data(self, ids=None, staff="upper", common_id=None):
        """
        Data structure suitable for passing to the PyPI krippendorff module.
        :param ids:
        :param staff:
        :param common_id:
        :return:
        """
        if ids is None:
            ids = []
        ignore = self._note_indices_to_ignore(staff=staff, common_id=common_id)
        data = []
        for coder_id in ids:
            coder_codings = list()
            note_index = 0
            annot = self._abcd_header.annotation_by_id(identifier=coder_id)
            for hsd in annot.handed_strike_digits(staff=staff):
                if note_index in ignore:
                    pass
                elif hsd is None or hsd == 'x':
                    coder_codings.append(np.nan)
                else:
                    coder_codings.append(hsd)
                note_index += 1
            data.append(coder_codings)
        # pprint.pprint(data)
        return data

    def _staff_annotation_data(self, ids=None, staff="upper", lib="nltk", label="unigram", common_id=None):
        if ids is None:
            ids = []
        if staff not in ('upper', 'lower'):
            raise Exception("Annotation data collected one staff at a time.")
        if label == "bigram":
            if lib == 'nltk':
                data = self._nltk_bigram_staff_annotation_data(ids=ids, staff=staff, common_id=common_id)
            else:
                data = self._pypi_bigram_staff_annotation_data(ids=ids, staff=staff, common_id=common_id)
        else:
            if lib == 'nltk':
                data = self._nltk_staff_annotation_data(ids=ids, staff=staff, common_id=common_id)
            else:
                data = self._pypi_staff_annotation_data(ids=ids, staff=staff, common_id=common_id)
        return data

    def _annotation_data(self, ids=None, staff="both", lib="nltk", label='unigram', common_id=None):
        """
        Separated data structures suitable for passing to the NLTK or PyPI krippendorff modules.
        :param ids:
        :param staff:
        :param common_id:
        :return:
        """
        if ids is None:
            ids = []
        staff_data = dict()
        upper_data = None
        lower_data = None
        if staff == "upper" or staff == "both":
            upper_data = self._staff_annotation_data(ids=ids, staff="upper", lib=lib, label=label, common_id=common_id)
        if staff == "lower" or staff == "both":
            lower_data = self._staff_annotation_data(ids=ids, staff="lower", lib=lib, label=label, common_id=common_id)
        staff_data['upper'] = upper_data
        staff_data['lower'] = lower_data
        # pprint.pprint(data)
        return staff_data

    def alpha(self, ids=None, staff="upper", common_id=None, lib='nltk', label='bigram', distance=None):
        if ids is None:
            ids = []
        if staff not in ('upper', 'lower'):
            raise Exception("Alpha measure only applicable one staff at a time.")

        data = self._staff_annotation_data(ids=ids, staff=staff, lib=lib, label=label, common_id=common_id)
        if distance is None and label == "bigram":
            distance = DScore.bigram_label_distance

        if lib == 'nltk':
            if distance is None:
                distance = binary_distance
            annot_task = AnnotationTask(data=data, distance=distance)
            krip = annot_task.alpha()
        else:
            if distance is None:
                distance = 'nominal'
            krip = alpha(reliability_data=data, level_of_measurement=distance)

        return krip

    def pypi_alpha_old(self, ids=None, staff="both", common_id=None, label='bigram', distance=binary_distance):
        if ids is None:
            ids = []
        if staff not in ('upper', 'lower'):
            raise Exception("PyPI krippendorff alpha only applicable one staff at a time.")

        if label == 'bigram':
            data = self._bigram_reliability_data(ids=ids, staff=staff, common_id=common_id)
        else:
            data = self._reliability_data(ids=ids, staff=staff, common_id=common_id)

        value_domain = ['>1', '>2', '>3', '>4', '>5', '<1', '<2', '<3', '<4', '<5']
        krip = alpha(reliability_data=data, level_of_measurement='nominal', value_domain=value_domain)
        return krip

    def _is_fully_annotated(self, staff="both", indices=None):
        if indices is None:
            indices = []
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

    def is_fully_annotated(self, staff="both", indices=None):
        """
        :return: True iff a strike digit is assigned to every note in the score.
        """
        if indices is None:
            indices = []
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

    def annotation_count(self):
        anns = self._abcd_header.annotations()
        ann_count = len(anns)
        return ann_count

    def remove_annotations(self):
        ah = self._abcd_header
        ah.remove_annotations()

    def annotate(self, d_annotation):
        ah = self._abcd_header
        ah.append_annotation(d_annotation=d_annotation)

    def annotations(self):
        hdr = self._abcd_header
        if hdr:
            annots = hdr.annotations()
            return annots
        return []

    def annotation_by_index(self, index):
        hdr = self._abcd_header
        if hdr:
            annot = hdr.annotation_by_index(index=index)
            return annot
        return None

    def annotation_by_id(self, identifier):
        hdr = self._abcd_header
        if hdr:
            annot = hdr.annotation_by_id(identifier=identifier)
            return annot
        return None

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
