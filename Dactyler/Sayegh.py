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

import networkx as nx
import copy
from Dactyler import Dactyler, Constant


class Sayegh(Dactyler.TrainedDactyler):

    def __init__(self, smoother=None):
        super().__init__()
        self._training = {} # W' in the Sayegh paper.
        self._smoother = smoother

    def _learn_from_example(self, transition_counts, fingered_counts):
        for (midi_1, digit_1, midi_2, digit_2) in fingered_counts:
            fingered_count = fingered_counts[(midi_1, digit_1, midi_2, digit_2)]
            transition_count = transition_counts[(midi_1, midi_2)]
            self._training[(midi_1, digit_1, midi_2, digit_2)] += fingered_count / transition_count;

    def smoother(self, method=None):
        if method is None:
            return self._smoother
        else:
            self._smoother = method

    def smooth(self):
        if self._smoother:
            raise Exception("Smoothing not yet available.")

    def train(self, d_corpus, staff="both", segmenter=None, annotation_indices=[]):
        if staff == "both":
            staves = ["upper", "lower"]
        elif staff == "upper":
            staves = ["upper"]
        elif staff == "lower":
            staves = ["lower"]
        else:
            raise Exception("Bad staff specification")

        lo, hi = d_corpus.pitch_range()
        for from_midi in range(lo, hi + 1):
            for from_hand in ('>', '<'):
                for from_digit in range(1, 6):
                    from_finger = from_hand + str(from_digit)
                    for to_midi in range(lo, hi + 1):
                        for to_hand in ('>', '<'):
                            for to_digit in range(1, 6):
                                to_finger = to_hand + str(to_digit)
                                self._training[(from_midi, from_finger, to_midi, to_finger)] = 0

        score_count = d_corpus.score_count()
        for i in range(score_count):
            d_score = d_corpus.d_score_by_index(i)
            print("Training from {0} score".format(d_score.title()))
            if d_score.is_fully_annotated(indices=annotation_indices):
                hdr = d_score.abcd_header()
                annot_index = 0
                for annot in hdr.annotations():
                    if len(annotation_indices) > 0 and annot_index not in annotation_indices:
                        continue
                    annot_index += 1

                    for stave in staves:
                        print("Staff {0}".format(stave))
                        prior_midi = None
                        prior_digit = None
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
                                note_index += 1
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

        segments = d_part.orderly_note_stream_segments()
        advice_segments = []
        segment_index = 0
        for segment in segments:
            if len(segment) == 1:
                seg_first_digit = None
                seg_last_digit = None
                if segment_index == 0:
                    seg_first_digit = first_digit
                if segment_index == len(segments) - 1:
                    seg_last_digit = last_digit
                advice = Dactyler.Dactyler.one_note_advice(segments[0], staff=staff,
                                                           first_digit=seg_first_digit,
                                                           last_digit=seg_last_digit)
                advice_segments.append(advice)
                continue

            g = nx.MultiDiGraph()
            g.add_node(0, midi=None, handed_digit=None)
            prior_slice_node_data = dict()
            prior_slice_node_data[0] = None
            node_id = 1
            for note in segment:
                path_exists = False
                slice_node_data = dict()
                for hand in ('>', '<'):
                    for digit in range(1, 6):
                        handed_digit = hand + str(digit)
                        g.add_node(node_id, midi=note.pitch.midi, handed_digit=handed_digit)
                        slice_node_data[node_id] = {"midi": note.pitch.midi, "handed_digit": handed_digit}
                        if 0 in prior_slice_node_data:
                            g.add_edge(0, node_id, weight=1)
                            path_exists = True
                        else:
                            for prior_node_id in prior_slice_node_data:
                                prior_midi = prior_slice_node_data[prior_node_id]["midi"]
                                prior_hd = prior_slice_node_data[prior_node_id]["handed_digit"]
                                weight_key = (prior_midi, prior_hd, note.pitch.midi, handed_digit)
                                # We will be searching for the shortest path over DAG, so the
                                # weights of favorable steps must be made small (negative).
                                weight = -1 * self._training[weight_key]
                                if weight < 0:
                                    # print("Weight for {0}: {1}".format(weight_key, weight))
                                    g.add_edge(prior_node_id, node_id, weight=weight)
                                    path_exists = True
                        node_id += 1
                if not path_exists:
                    raise Exception("Model insufficiently trained to find solution. Train more or try smoothing.")
                prior_slice_node_data = copy.copy(slice_node_data)
                segment_index += 1
            g.add_node(node_id, midi=None, handed_digit=None)
            for prior_node_id in prior_slice_node_data:
                g.add_edge(prior_node_id, node_id, weight=1)

            path = nx.shortest_path(g, source=0, target=node_id, weight="weight")
            segment_abcdf = ''
            for node_id in path:
                node = g.nodes[node_id]
                if node["handed_digit"]:
                    segment_abcdf += node["handed_digit"]
            advice_segments.append(segment_abcdf)

        abcdf = Dactyler.Dactyler.combine_abcdf_segments(advice_segments)
        return abcdf
