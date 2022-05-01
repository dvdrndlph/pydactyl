#!/usr/bin/env python
__author__ = 'David Randolph'
# Copyright (c) 2020-2021 David A. Randolph.
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
import pprint
import re
import copy
import sys
import time

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn_crfsuite import scorers

import sklearn_crfsuite as crf
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from pydactyl.eval.Corporeal import Corporeal
from pydactyl.dactyler.Parncutt import TrigramNode
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter

VERSION = '0000'
MAX_LEAP = 16
MICROSECONDS_PER_BEAT = 500000
MS_PER_BEAT = MICROSECONDS_PER_BEAT / 1000
CHORD_MS_THRESHOLD = 30
# CROSS_VALIDATE = False
# One of 'cross-validate', 'preset', 'random'
TEST_METHOD = 'cross-validate'
TEST_METHOD = 'preset'
# TEST_METHOD = 'random'
SEGREGATE_HANDS = False
STAFFS = ['upper', 'lower']
# STAFFS = ['upper']
# CORPUS_NAMES = ['full_american_by_annotator']
# CORPUS_NAMES = ['layer_one_by_annotator']
# CORPUS_NAMES = ['scales']
# CORPUS_NAMES = ['arpeggios']
# CORPUS_NAMES = ['broken']
# CORPUS_NAMES = ['layer_one_by_annotator', 'scales', 'arpeggios', 'broken']
# CORPUS_NAMES = ['scales', 'arpeggios', 'broken']
# CORPUS_NAMES = ['pig']
CORPUS_NAMES = ['pig_indy']


#####################################################
# FUNCTIONS
#####################################################
def is_in_test_set(title: str, corpus='pig_indy'):
    if corpus in ('pig_indy', 'pig'):
        example, annotator_id = title.split('_')
        example_int = int(example)
        if example_int <= 30:
            return True
    else:
        raise Exception("Not implemented yet.")
    return False

def get_trigram_node(notes, annotations, i):
    midi_1 = None
    handed_digit_1 = '-'
    if i > 0:
        midi_1 = notes[i-1].midi()
        handed_digit_1 = annotations[i-1]
    midi_2 = notes[i].midi()
    handed_digit_2 = annotations[i]

    midi_3 = None
    handed_digit_3 = '-'
    if i < len(notes) - 1:
        midi_3 = notes[i+1].midi()
        handed_digit_3 = annotations[i+1]
    trigram_node = TrigramNode(midi_1=midi_1, handed_digit_1=handed_digit_1,
                               midi_2=midi_2, handed_digit_2=handed_digit_2,
                               midi_3=midi_3, handed_digit_3=handed_digit_3)
    return trigram_node


def leap_is_excessive(notes, middle_i):
    left_i = middle_i - 1
    if left_i in notes:
        leap = notes[middle_i].pitch.midi - notes[left_i].pitch.midi
        if abs(leap) > MAX_LEAP:
            return True
    else:
        return True  # That first step is a doozy. Infinite leap.
    return False


def chordings(stream, middle_i):
    middle_offset_beats = stream[middle_i].offset
    middle_offset_ms = middle_offset_beats * MS_PER_BEAT
    min_left_offset_ms = middle_offset_ms - CHORD_MS_THRESHOLD
    max_right_offset_ms = middle_offset_ms + CHORD_MS_THRESHOLD
    # Notes (other than the leftmost--lowest--note) in chords with identical
    # onset times in the performance have been shifted right by a 1/2048th
    # note duration to provide a total-order to coincide with the ABCDF fingering
    # sequence.
    left_chord_notes = 0
    for i in range(middle_i, middle_i - 6, -1):
        if i < 0:
            break
        i_offet_ms = stream[i].offset * MS_PER_BEAT
        if i_offet_ms > min_left_offset_ms:
            left_chord_notes += 1
    right_chord_notes = 0
    for i in range(middle_i, middle_i + 6, 1):
        if i >= len(stream):
            break
        i_offet_ms = stream[i].offset * MS_PER_BEAT
        if i_offet_ms < max_right_offset_ms:
            right_chord_notes += 1
    return left_chord_notes, right_chord_notes


def note2features(notes, stream, annotations, i, staff):
    trigram_node = get_trigram_node(notes, annotations, i)
    features = {}
    functions = judge.rules()
    for tag, rule_method in functions.items():
        raw_cost = rule_method(trigram_node)
        features[tag] = raw_cost

    features['staff'] = 0
    if staff == "upper":
        features[staff] = 1
        # @100: [0.54495717 0.81059147 0.81998371 0.68739401 0.73993751]
        # @1:   [0.54408935 0.80563961 0.82079826 0.6941775  0.73534277]

    # FIXME: Add chord features. Approximate with 30 ms offset deltas.
    left_chord_notes, right_chord_notes = chordings(stream=stream, middle_i=i)
    features['left_chord'] = left_chord_notes
    features['right_chord'] = right_chord_notes

    # FIXME: Impact of large leaps? Costs max out, no? Maybe not.
    # features['leap'] = 0
    # if leap_is_excessive(notes, i):
    # features['leap'] = 1

    # FIXME: Lattice distance in Parncutt rules? Approximated by Jacobs.
    #        Mitigated by Balliauw (which just makes the x-distance more
    #        accurate between same-colored keys).
    # FIXME: Articulation (legato, staccato)?
    # FIXME: tempo (window duration)?

    return features


def phrase2features(notes, stream, annotations, staff):
    feature_list = []
    for i in range(len(notes)):
        features = note2features(notes, stream, annotations, i, staff)
        feature_list.append(features)
    return feature_list


def phrase2labels(handed_strike_digits):
    return handed_strike_digits


def phrase2tokens(notes):
    tokens = []
    for d_note in notes:
        m21_note = d_note.m21_note()
        nom = m21_note.nameWithOctave
        tokens.append(nom)
    return tokens


def nondefault_hand_count(hsd_seq, staff="upper"):
    nondefault_hand = '<'
    if staff == 'lower':
        nondefault_hand = '>'
    bad_hand_cnt = 0
    for fingering in hsd_seq:
        if fingering[0] == nondefault_hand:
            bad_hand_cnt += 1
    return bad_hand_cnt


def has_wildcard(hsd_seq):
    for fingering in hsd_seq:
        if fingering[0] == 'x':
            return True
    return False


def train_and_evaluate(the_model, x_train, y_train, x_test, y_test):
    the_model.fit(x_train, y_train)
    labels = list(my_crf.classes_)
    print(labels)

    y_predicted = my_crf.predict(x_test)
    print("Predicted: {}".format(y_predicted))

    flat_f1 = metrics.flat_f1_score(y_test, y_predicted, average='weighted', labels=labels)
    print("Flat F1: {}".format(flat_f1))

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(y_test, y_predicted, labels=sorted_labels, digits=3))


#####################################################
# MAIN BLOCK
#####################################################
creal = Corporeal()
judge = creal.get_model('parncutt')
# judge = creal.get_model('badball')
# judge = creal.get_model('jacobs')

# token_lists = []
x = []
y = []
x_train = []
x_test = []
y_train = []
y_test = []

bad_annot_count = 0
wildcarded_count = 0
good_annot_count = 0
included_note_count = 0
total_nondefault_hand_finger_count = 0
total_nondefault_hand_segment_count = 0
for corpus_name in CORPUS_NAMES:
    da_corpus = creal.get_corpus(corpus_name=corpus_name)
    for da_score in da_corpus.d_score_list():
        for staff in STAFFS:
            score_title = da_score.title()
            # if score_title != 'Sonatina 4.1':
                # continue
            print("Processing {} staff of {} score from {} corpus.".format(staff, score_title, corpus_name))
            abcdh = da_score.abcd_header()
            annot_count = abcdh.annotation_count()
            annot = da_score.annotation_by_index(index=0)
            segger = ManualDSegmenter(level='.', d_annotation=annot)
            da_score.segmenter(segger)
            # FIXME: Calling both of the following is probably not necessary...
            orderly_stream_segments = da_score.orderly_note_stream_segments(staff=staff)
            orderly_note_segments = da_score.orderly_d_note_segments(staff=staff)
            seg_count = len(orderly_note_segments)
            for annot_index in range(annot_count):
                annot = da_score.annotation_by_index(annot_index)
                authority = annot.authority()
                hsd_segments = segger.segment_annotation(annotation=annot, staff=staff)
                seg_index = 0
                for hsd_seg in hsd_segments:
                    ordered_notes = orderly_note_segments[seg_index]
                    ordered_stream = orderly_stream_segments[seg_index]
                    # token_lists.append(phrase2tokens(ordered_notes))
                    note_len = len(ordered_notes)
                    seg_len = len(hsd_seg)
                    if note_len != seg_len:
                        print("Bad annotation by {} for score {}. Notes: {} Fingers: {}".format(
                            authority, score_title, note_len, seg_len))
                        bad_annot_count += 1
                        continue
                    nondefault_hand_finger_count = nondefault_hand_count(hsd_seq=hsd_seg, staff=staff)
                    if nondefault_hand_finger_count:
                        total_nondefault_hand_segment_count += 1
                        print("Non-default hand specified vy annotator {} in score {}: {}".format(
                            authority, score_title, hsd_seg))
                        total_nondefault_hand_finger_count += nondefault_hand_finger_count
                        if SEGREGATE_HANDS:
                            bad_annot_count += 1
                            continue
                    if has_wildcard(hsd_seq=hsd_seg):
                        # print("Wildcard disallowed from annotator {} in score {}: {}".format(
                            # authority, score_title, hsd_seg))
                        wildcarded_count += 1
                        continue
                    included_note_count += note_len
                    x.append(phrase2features(ordered_notes, ordered_stream, hsd_seg, staff))
                    y.append(phrase2labels(hsd_seg))
                    if TEST_METHOD == 'preset':
                        if is_in_test_set(title=score_title, corpus=corpus_name):
                            x_test.append(phrase2features(ordered_notes, ordered_stream, hsd_seg, staff))
                            y_test.append(phrase2labels(hsd_seg))
                        else:
                            x_train.append(phrase2features(ordered_notes, ordered_stream, hsd_seg, staff))
                            y_train.append(phrase2labels(hsd_seg))

                    good_annot_count += 1

# print(token_lists)
print("Example count: {}".format(len(x)))
if TEST_METHOD == 'preset':
    print("Training count: {}".format(len(y_train)))
    print("Test count: {}".format(len(y_test)))
print("Good examples: {}".format(good_annot_count))
print("Bad examples: {}".format(bad_annot_count))
print("Wildcarded examples: {}".format(wildcarded_count))
print("Total notes included: {}".format(included_note_count))
print("Total nondefault hand fingerings: {}".format(total_nondefault_hand_finger_count))
print("Total nondefault hand phrases: {}".format(total_nondefault_hand_segment_count))

my_crf = crf.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    # max_iterations=100,
    all_possible_transitions=True
)
if TEST_METHOD == 'cross-validate':
    scores = cross_val_score(my_crf, x, y, cv=5)
    # scores = cross_validate(my_crf, x, y, cv=5, scoring="flat_precision_score")
    print(scores)
elif TEST_METHOD == 'preset':
    my_crf.fit(x_train, y_train)
    train_and_evaluate(the_model=my_crf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    train_and_evaluate(the_model=my_crf, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
