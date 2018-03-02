__author__ = 'David Randolph'
# Copyright (c) 2018 David A. Randolph.
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
#
""" Here we migrate a method originally presented for guitar to piano: 

        S. I. Sayegh, “Fingering for string instruments with the optimum
            path paradigm,” Comput. Music J., vol. 13, no. 3, pp. 76–84, 1989.
            
    A nugget cuz we dug it.
"""
import numpy
import networkx
import re
from Dactyler import Dactyler, Constant


class Sayegh(Dactyler.TrainedDactyler):

    def __init__(self):
        super().__init__()
        self._training = {} # W' in the Sayegh paper.

    def _learn_from_example(self, transition_counts, fingered_counts):
        for (midi_1, digit_1, midi_2, digit_2) in fingered_counts:
            fingered_count = fingered_counts[(midi_1, digit_1, midi_2, digit_2)]
            transition_count = transition_counts[(midi_1, midi_2)]
            self._training[(midi_1, digit_1, midi_2, digit_2)] += fingered_count / transition_count;

    def train(self, d_corpus, staff="both", segmenter=None, annotation_indices=[]):
        if staff == "both":
            staves = ["upper", "lower"]
        elif staff == "upper":
            staves = ["upper"]
        elif staff == "lower":
            staves = ["lower"]
        else:
            raise Exception("Bad staff specification")

        self._weights = {}
        lo, hi = d_corpus.pitch_range()
        for from_midi in range(lo, hi + 1):
            for from_hand in ('>', '<'):
                for from_digit in range(1, 6):
                    from_finger = from_hand + str(from_digit)
                    for to_midi in range(lo, hi + 1):
                        for to_hand in ('<', '>'):
                            for to_digit in range(1, 6):
                                to_finger = to_hand + str(to_digit)
                                self._training[(from_midi, from_finger, to_midi, to_finger)] = 0

        score_count = d_corpus.score_count()
        for i in range(score_count):
            d_score = d_corpus.d_score_by_index(i)
            if d_score.is_fully_annotated(indices=annotation_indices):
                hdr = d_score.abcd_header()
                annot_index = 0
                prior_midi = None
                prior_digit = None
                for annot in hdr.annotations():
                    if len(annotation_indices) > 0 and annot_index not in annotation_indices:
                        continue
                    annot_index += 1

                    for stave in staves:
                        transition_counts = {}
                        fingered_counts = {}
                        handed_digits = annot.handed_strike_digits(staff=stave)
                        stave_notes = d_score.orderly_note_stream(staff=stave)
                        stave_segments = d_score.orderly_note_stream_segments(staff=stave)
                        if len(handed_digits) != len(stave_notes):
                            raise Exception("Strike digit mismatch in {0}: {1} notes and {2} digits".format(
                                d_score.title(), len(stave_notes), len(handed_digits)))

                        sf_count = annot.score_fingering_count(staff=stave)
                        segment_index = 0
                        note_index = 0
                        segment = stave_segments[segment_index]
                        for j in range(sf_count):
                            if note_index == len(segment):
                                self._learn_from_example(transition_counts=transition_counts,
                                                         fingered_counts=fingered_counts)
                                segment_index += 1
                                note_index = 0
                            current_note = stave_segments[segment_index][note_index]
                            current_midi = current_note.pitch.midi
                            current_digit = handed_digits[j]
                            if prior_midi is None:
                                prior_midi = current_midi
                                prior_digit = current_digit
                                continue
                            print("{0} {1} --> {2} {3}".format(prior_midi, prior_digit, current_midi, current_digit))
                            if (prior_midi, current_midi) in transition_counts:
                                transition_counts[(prior_midi, current_midi)] += 1
                            else:
                                transition_counts[(prior_midi, current_midi)] = 1
                            if (prior_midi, prior_digit, current_midi, current_digit) in fingered_counts:
                                fingered_counts[(prior_midi, prior_digit, current_midi, current_digit)] += 1
                            else:
                                fingered_counts[(prior_midi, prior_digit, current_midi, current_digit)] = 1

                            prior_midi = current_midi
                            prior_digit = current_digit
                            note_index += 1

                        self._learn_from_example(transition_counts=transition_counts,
                                                 fingered_counts=fingered_counts)

    def advise(self, score_index=0, staff="upper", offset=0, first_digit=None, last_digit=None):
        d_scores = self._d_corpus.d_score_list()
        if score_index >= len(d_scores):
            raise Exception("Score index out of range")

        d_score = d_scores[score_index]
        if staff == "both":
            upper_advice = self.advise(score_index=score_index, staff="upper")
            abcdf = upper_advice + "@"
            if d_score.part_count() > 1:
                lower_advice = self.advise(score_index=score_index, staff="lower")
                abcdf += lower_advice
            return abcdf

        if staff != "upper" and staff != "lower":
            raise Exception("Segregated advice is only dispensed one staff at a time.")

        if d_score.part_count() == 1:
            d_part = d_score.combined_d_part()
        else:
            # We support (segregated) left hand fingerings. By segregated, we
            # mean the right hand is dedicated to the upper staff, and the
            # left hand is dedicated to the lower staff.
            d_part = d_score.d_part(staff=staff)

        m21_stream = d_part.orderly_note_stream()