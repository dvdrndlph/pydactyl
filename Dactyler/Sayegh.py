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
import re
import copy
from Dactyler import Dactyler as D


class Sayegh(D.TrainedDactyler):

    def __init__(self, smoother=None):
        super().__init__()
        self._training = dict()  # W' in the Sayegh paper.
        self._smoother = smoother

    @staticmethod
    def _learn_from_example(staff, transition_counts, fingered_counts, training):
        for (midi_1, midi_2, digit_1, digit_2) in fingered_counts:
            fingered_count = fingered_counts[(midi_1, midi_2, digit_1, digit_2)]
            transition_count = transition_counts[(midi_1, midi_2)]
            training[(midi_1, midi_2, digit_1, digit_2)] += fingered_count / transition_count;

    def _paths(self, staff):
        path_exists = dict()
        for (from_midi, to_midi, from_finger, to_finger) in self._training[staff]:
            weight = self._training[staff][(from_midi, to_midi, from_finger, to_finger)]
            if weight > 0:
                path_exists[(from_midi, to_midi)] = True
        return path_exists

    def smooth(self):
        """
            For "uniform" smoothing, we "smooth" by finding all note transitions which are not
            represented in the training data and giving all fingering transitions a uniform
            non-zero weight.
        """
        for staff in ('upper', 'lower'):
            existing_paths = self._paths(staff=staff)
            if self._smoother in "uniform":
                for (from_midi, to_midi, from_finger, to_finger) in self._training[staff]:
                    if (from_midi, to_midi) in existing_paths:
                        continue
                self._training[staff][(from_midi, to_midi, from_finger, to_finger)] = 0.0001

    def train(self, d_corpus, staff="both", segregate=True, segmenter=None, annotation_indices=[]):
        if not segregate:
            raise Exception("Desegregated (integrated) fingering not supported.")

        if staff == "both":
            self.train(d_corpus, staff="upper", segmenter=segmenter, annotation_indices=annotation_indices)
            self.train(d_corpus, staff="lower", segmenter=segmenter, annotation_indices=annotation_indices)
            return

        training = dict()
        lo, hi = d_corpus.pitch_range(staff=staff)
        hand = ">"
        if staff == "lower":
            hand = "<"
        for from_midi in range(lo, hi + 1):
            for from_digit in range(1, 6):
                from_finger = hand + str(from_digit)
                for to_midi in range(lo, hi + 1):
                    for to_digit in range(1, 6):
                        to_finger = hand + str(to_digit)
                        training[(from_midi, to_midi, from_finger, to_finger)] = 0

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

                    prior_midi = None
                    prior_digit = None
                    transition_counts = {}
                    fingered_counts = {}
                    handed_digits = annot.handed_strike_digits(staff=staff)
                    stave_notes = d_score.orderly_note_stream(staff=staff)
                    stave_segments = d_score.orderly_note_stream_segments(staff=staff)
                    if len(handed_digits) != len(stave_notes):
                        raise Exception("Strike digit mismatch in {0}: {1} notes and {2} digits".format(
                            d_score.title(), len(stave_notes), len(handed_digits)))

                    sf_count = annot.score_fingering_count(staff=staff)
                    segment_index = 0
                    note_index = 0
                    segment = stave_segments[segment_index]
                    for j in range(sf_count):
                        if note_index == len(segment):
                            Sayegh._learn_from_example(staff=staff,
                                                       transition_counts=transition_counts,
                                                       fingered_counts=fingered_counts,
                                                       training=training)
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
                            fingered_counts[(prior_midi, current_midi, prior_digit, current_digit)] += 1
                        else:
                            fingered_counts[(prior_midi, current_midi, prior_digit, current_digit)] = 1

                        prior_midi = current_midi
                        prior_digit = current_digit
                        note_index += 1

                    Sayegh._learn_from_example(staff=staff,
                                               transition_counts=transition_counts,
                                               fingered_counts=fingered_counts,
                                               training=training)


    def advise(self, score_index=0, staff="upper", offset=0, first_digit=None, last_digit=None, top=None):
        print("STAFF: {0} FIRST: {1} LAST: {2}".format(staff, first_digit, last_digit))
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

        handed_first_digit = D.Dactyler.hand_digit(digit=first_digit, staff=staff)
        print("First digit: {0} Handed first digit: {1}".format(first_digit, handed_first_digit))
        handed_last_digit = D.Dactyler.hand_digit(digit=last_digit, staff=staff)
        print("Last digit: {0} Handed last digit: {1}".format(last_digit, handed_last_digit))

        if d_score.part_count() == 1:
            d_part = d_score.combined_d_part()
        else:
            # We support (segregated) left hand fingerings. By segregated, we
            # mean the right hand is dedicated to the upper staff, and the
            # left hand is dedicated to the lower staff.
            d_part = d_score.d_part(staff=staff)

        lo, hi = d_part.pitch_range()
        print("Pitches {0} to {1}".format(lo, hi))

        hand = ">"
        if staff == "lower":
            hand = "<"

        segments = d_part.orderly_note_stream_segments(offset=offset)
        advice_segments = []
        segment_index = 0
        last_segment_index = len(segments) - 1
        for segment in segments:
            if len(segment) == 1:
                d_note = D.DNote(segment[0])
                advice = D.Dactyler.one_note_advice(d_note, staff=staff,
                                                    first_digit=handed_first_digit,
                                                    last_digit=handed_last_digit)
                advice_segments.append(advice)
                continue

            g = nx.MultiDiGraph()
            g.add_node(0, midi=None, handed_digit=None)
            prior_slice_node_ids = list()
            prior_slice_node_ids.append(0)
            last_note_in_segment_index = len(segment) - 1
            note_in_segment_index = 0
            node_id = 1
            on_last_prefingered_note = False
            for note in segment:
                on_first_prefingered_note = False
                slice_node_ids = list()

                if segment_index == 0 and note_in_segment_index == 0 and handed_first_digit:
                    on_first_prefingered_note = True

                if segment_index == last_segment_index and \
                        note_in_segment_index == last_note_in_segment_index and handed_last_digit:
                    on_last_prefingered_note = True

                for digit in range(1, 6):
                    handed_digit = hand + str(digit)
                    if on_last_prefingered_note and handed_digit != handed_last_digit:
                        continue
                    if on_first_prefingered_note and handed_digit != handed_first_digit:
                        continue
                    g.add_node(node_id, midi=note.pitch.midi, handed_digit=handed_digit)
                    slice_node_ids.append(node_id)
                    if 0 in prior_slice_node_ids:
                        g.add_edge(0, node_id, weight=-1)
                    else:
                        for prior_node_id in prior_slice_node_ids:
                            prior_node = g.nodes[prior_node_id]
                            prior_midi = prior_node["midi"]
                            prior_hd = prior_node["handed_digit"]
                            weight_key = (prior_midi, note.pitch.midi, prior_hd, handed_digit)
                            # We will be searching for the shortest path over DAG, so the
                            # weights of favorable steps must be made small (negative).
                            weight = -1 * self._training[staff][weight_key]
                            g.add_edge(prior_node_id, node_id, weight=weight)
                    node_id += 1
                if len(slice_node_ids) > 0:
                    prior_slice_node_ids = copy.copy(slice_node_ids)
                note_in_segment_index += 1

            g.add_node(node_id, midi=None, handed_digit=None)
            for prior_node_id in prior_slice_node_ids:
                g.add_edge(prior_node_id, node_id, weight=-1)

            path = nx.shortest_path(g, source=0, target=node_id, weight="weight")
            segment_abcdf = ''
            for node_id in path:
                node = g.nodes[node_id]
                if node["handed_digit"]:
                    segment_abcdf += node["handed_digit"]
            advice_segments.append(segment_abcdf)
            segment_index += 1

        abcdf = D.Dactyler.combine_abcdf_segments(advice_segments)
        return abcdf