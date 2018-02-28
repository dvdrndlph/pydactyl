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

import numpy
import networkx
import re
from Dactyler import Dactyler, Constant


class Sayegh(Dactyler.TrainedDactyler):

    def __init__(self):
        super().__init__()

    def train(self, d_corpus, staff="both", annotation_indices=[]):
        if staff == "both":
            staves = ["upper", "lower"]
        elif staff == "upper":
            staves = ["upper"]
        elif staff == "lower":
            staves = ["lower"]
        else:
            raise Exception("Bad staff specification")

        lo, hi = d_corpus.pitch_range()
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
                        handed_digits = annot.handed_strike_digits(staff=stave)
                        stave_notes = d_score.orderly_note_stream(staff=stave)
                        if len(handed_digits) != len(stave_notes):
                            raise Exception("Strike digit mismatch in {0}: {1} notes and {2} digits".format(
                                d_score.title(), len(stave_notes), len(handed_digits)))
                        sf_count = annot.score_fingering_count(staff=stave)
                        for j in range(sf_count):
                            current_midi = stave_notes[j].pitch.midi
                            current_digit = handed_digits[j]
                            if prior_midi is None:
                                prior_midi = current_midi
                                prior_digit = current_digit
                                continue
                            print("{0} {1} --> {2} {3}".format(prior_midi, prior_digit, current_midi, current_digit))
                            prior_midi = current_midi
                            prior_digit = current_digit

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