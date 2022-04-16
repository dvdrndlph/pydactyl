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
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from pydactyl.eval.Corporeal import Corporeal, RANK_HEADINGS, ERR_HEADINGS, ERR_METHODS, WEIGHT_RE, STAFF
from pydactyl.dcorpus.PianoFingering import PianoFingering
from pydactyl.dcorpus.DCorpus import DCorpus, DScore, DAnnotation
from pydactyl.dactyler.Parncutt import Parncutt, TrigramNode, Badgerow, Balliauw, Jacobs
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter

VERSION = '0000'
CORPUS_NAMES = ['full_american_by_annotator']
CORPUS_NAMES = ['layer_one_by_annotator']
CORPUS_NAMES = ['scales']
CROSS_VALIDATE = False
CROSS_VALIDATE = True
# CORPUS_NAMES = ['arpeggios']
# CORPUS_NAMES = ['broken']
CORPUS_NAMES = ['layer_one_by_annotator', 'scales', 'arpeggios', 'broken']
# CORPUS_NAMES = ['scales', 'arpeggios', 'broken']


#####################################################
# FUNCTIONS
#####################################################
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


def note2features(notes, annotations, i):
    trigram_node = get_trigram_node(notes, annotations, i)
    # FIXME: Chord features?
    features = {}
    functions = judge.rules()
    for tag, rule_method in functions.items():
        raw_cost = rule_method(trigram_node)
        features[tag] = raw_cost

    return features


def phrase2features(notes, annotations):
    feature_list = []
    for i in range(len(notes)):
        features = note2features(notes, annotations, i)
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


def has_left_hand(hsd_seq):
    for fingering in hsd_seq:
        if fingering[0] == '<':
            return True
    return False


def has_wildcard(hsd_seq):
    for fingering in hsd_seq:
        if fingering[0] == 'x':
            return True
    return False


#####################################################
# MAIN BLOCK
#####################################################
creal = Corporeal()
judge = Parncutt()

# token_lists = []
x = []
y = []
test_annot_id = 3
bad_annot_count = 0
wildcarded_count = 0

good_annot_count = 0
for corpus_name in CORPUS_NAMES:
    da_corpus = creal.get_corpus(corpus_name=corpus_name)
    for da_score in da_corpus.d_score_list():
        abcdh = da_score.abcd_header()
        annot_count = abcdh.annotation_count()
        annot = da_score.annotation_by_id(identifier=1)
        segger = ManualDSegmenter(level='.', d_annotation=annot)
        da_score.segmenter(segger)
        score_title = da_score.title()
        orderly_note_segments = da_score.orderly_d_note_segments(staff=STAFF)
        seg_count = len(orderly_note_segments)
        for annot_id in range(1, annot_count + 1, 1):
            annot = da_score.annotation_by_id(identifier=annot_id)
            authority = annot.authority()
            hsd_segments = segger.segment_annotation(annotation=annot, staff=STAFF)
            print(hsd_segments)
            seg_index = 0
            for hsd_seg in hsd_segments:
                ordered_notes = orderly_note_segments[seg_index]
                # token_lists.append(phrase2tokens(ordered_notes))
                note_len = len(ordered_notes)
                seg_len = len(hsd_seg)
                if note_len != seg_len:
                    print("Bad annotation. Notes: {} Fingers: {}".format(note_len, seg_len))
                    print(da_score)
                    print(annot)
                    raise Exception("Bad")
                if STAFF == "upper":
                    if has_left_hand(hsd_seq=hsd_seg):
                        print("Left hand disallowed by annotator {} in score {}: {}".format(
                            authority, score_title, hsd_seg))
                        bad_annot_count += 1
                        continue
                if has_wildcard(hsd_seq=hsd_seg):
                    print("Wildcard disallowed by annotator {} in score {}: {}".format(
                        authority, score_title, hsd_seg))
                    wildcarded_count += 1
                    continue
                x.append(phrase2features(ordered_notes, hsd_seg))
                y.append(phrase2labels(hsd_seg))
                good_annot_count += 1

# print(token_lists)
print("Example count: {}".format(len(x)))
print("Training count: {}".format(len(y)))
print("Good examples: {}".format(good_annot_count))
print("Bad examples: {}".format(bad_annot_count))
print("Wildcarded examples: {}".format(wildcarded_count))

my_crf = crf.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
if CROSS_VALIDATE:
    scores = cross_val_score(my_crf, x, y, cv=5)
    # scores = cross_validate(my_crf, x, y, cv=5, scoring="flat_precision_score")
    print(scores)
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    my_crf.fit(x_train, y_train)
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
