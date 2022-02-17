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
from pydactyl.eval.Corporeal import Corporeal, RANK_HEADINGS, ERR_HEADINGS, ERR_METHODS, WEIGHT_RE, STAFF
from pydactyl.dcorpus.PianoFingering import PianoFingering
from pydactyl.dcorpus.DCorpus import DCorpus, DScore, DAnnotation

VERSION = '0000'
CORPUS_NAMES = ['full_american_by_annotator']

#####################################################
# FUNCTIONS
#####################################################
def note2features(notes, i):
    note = notes[i]
    # FIXME: Chord features?
    features = {
        'bias': 1.0,
        'midi': note.midi(),
        'black': note.is_black(),
        'cramped': note.is_cramped(),
        'first': True,
        'last': True,
        '-1+1:between': False,
        '-2+2:between': False,
        '-1:black': False,
        '-1:cramped': False,
        '-1:semitone_delta': 0,
        '-1:balliauw_delta': 0,
        '-2:black': False,
        '-2:cramped': False,
        '-2:semitone_delta': 0,
        '-2:balliauw_delta': 0,
        '+1:black': False,
        '+1:cramped': False,
        '+1:semitone_delta': 0,
        '+1:balliauw_delta': 0,
        '+2:black': False,
        '+2:cramped': False,
        '+2:semitone_delta': 0,
        '+2:balliauw_delta': 0
    }
    if i > 0:
        prior_note = notes[i-1]
        features.update({
            'first': False,
            '-1:black': prior_note.is_black(),
            '-1:cramped': prior_note.is_cramped(),
            '-1:semitone_delta': note.signed_semitone_delta(),
            '-1:balliauw_delta': note.signed_balliauw_delta(),
        })
    if i > 1:
        left_note = notes[i-2]
        features.update({
            '-2:black': left_note.is_black(),
            '-2:cramped': left_note.is_cramped(),
            '-2:semitone_delta': note.signed_semitone_delta_from(left_note),
            '-2:balliauw_delta': note.signed_balliauw_delta_from(left_note),
        })
    if i < len(notes) - 1:
        next_note = notes[i+1]
        features.update({
            'last': False,
            '+1:black': next_note.is_black(),
            '+1:cramped': next_note.is_cramped(),
            '+1:semitone_delta': next_note.signed_semitone_delta(),
            '+1:balliauw_delta': next_note.signed_balliauw_delta()
        })
    if i < len(notes) - 2:
        right_note = notes[i+1]
        features.update({
            '+2:black': right_note.is_black(),
            '+2:cramped': right_note.is_cramped(),
            '+2:semitone_delta': right_note.signed_semitone_delta_from(note),
            '+2:balliauw_delta': right_note.signed_balliauw_delta_from(note)
        })
    if 0 < i < len(notes) - 1:
        prior_note = notes[i-1]
        next_note = notes[i+1]
        features['-1+1:between'] = note.is_between(prior_note, next_note)
    if 1 < i < len(notes) - 2:
        left_note = notes[i-2]
        right_note = notes[i+2]
        features['-2+2:between'] = note.is_between(left_note, right_note)

    return features


def phrase2features(notes):
    feature_list = []
    for i in range(len(notes)):
        features = note2features(notes, i)
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


#####################################################
# MAIN BLOCK
#####################################################
creal = Corporeal()
file_mode = 'a'

token_lists = []
x_train = []
y_train = []
x_test = []
y_test = []
test_annot_id = 1
for corpus_name in CORPUS_NAMES:
    da_corpus = creal.get_corpus(corpus_name=corpus_name)
    for da_score in da_corpus.d_score_list():
        ordered_notes = da_score.orderly_d_notes(staff=STAFF)
        # print(ordered_notes)
        token_lists.append(phrase2tokens(ordered_notes))
        abcdh = da_score.abcd_header()
        annot_count = abcdh.annotation_count()
        for annot_id in range(1, annot_count + 1, 1):
            annot = da_score.annotation_by_id(identifier=annot_id)
            hsd = annot.handed_strike_digits(staff=STAFF)
            if len(ordered_notes) != len(hsd):
                print("Bad annotation. Notes: {} Fingers: {}".format(len(ordered_notes), len(hsd)))
                print(da_score)
                print(annot)
                raise Exception("Bad")
            if annot_id == test_annot_id:
                x_test.append(phrase2features(ordered_notes))
                y_test.append(phrase2labels(hsd))
            else:
                x_train.append(phrase2features(ordered_notes))
                y_train.append(phrase2labels(hsd))

print(token_lists)
print(len(x_train))
print(len(y_train))

print(len(x_test))
print(len(y_test))


crf = crf.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(x_train, y_train)
labels = list(crf.classes_)
print(labels)

y_predicted = crf.predict(x_test)
print("Predicted: {}".format(y_predicted))

flat_f1 = metrics.flat_f1_score(y_test, y_predicted, average='weighted', labels=labels)
print("Flat F1: {}".format(flat_f1))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(y_test, y_predicted, labels=sorted_labels, digits=3))
