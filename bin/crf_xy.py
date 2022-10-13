#!/usr/bin/env python
__author__ = 'David Randolph'

import copy

# Copyright (c) 2020-2022 David A. Randolph.
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
# Second-order linear chain CRF piano fingering models, implemented using PySeqLab,
# which does not seem to provide a way to predefine "edge-observation" functions
# over both observations and labels.
#
# from pyseqlab.utilities import SequenceStruct
import sklearn_crfsuite as crf
from sklearn_crfsuite import metrics
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from pydactyl.util.DExperiment import DExperiment
from pydactyl.util.XyCrf import XyCrf, START_TAG, STOP_TAG
import pydactyl.util.CrfUtil as c
from pydactyl.eval.Corporeal import Corporeal
from pydactyl.dactyler.Parncutt import TrigramNode


# CROSS_VALIDATE = False
# One of 'cross-validate', 'preset', 'random'
# TEST_METHOD = 'cross-validate'
TEST_METHOD = 'preset'
# TEST_METHOD = 'random'
STAFFS = ['upper', 'lower']
# STAFFS = ['upper']
# CORPUS_NAMES = ['full_american_by_annotator']
# CORPUS_NAMES = ['complete_layer_one']
# CORPUS_NAMES = ['scales']
# CORPUS_NAMES = ['arpeggios']
# CORPUS_NAMES = ['broken']
# CORPUS_NAMES = ['complete_layer_one', 'scales', 'arpeggios', 'broken']
CORPUS_NAMES = ['scales', 'arpeggios', 'broken']
# CORPUS_NAMES = ['pig']
# CORPUS_NAMES = ['pig_indy']
# CORPUS_NAMES = ['pig_seg']


#####################################################
# FUNCTIONS
#####################################################
def get_trigram_node(notes, i, y_prev, y):
    midi_1 = None
    handed_digit_1 = '-'
    if i > 0:
        midi_1 = notes[i-1]['note'].pitch.midi
        handed_digit_1 = y_prev
    midi_2 = notes[i]['note'].pitch.midi
    handed_digit_2 = y

    midi_3 = None
    handed_digit_3 = '-'
    if i < len(notes) - 1:
        midi_3 = notes[i+1]['note'].pitch.midi
        # We don't know the subsequent finger, but some rules don't care.
    trigram_node = TrigramNode(midi_1=midi_1, handed_digit_1=handed_digit_1,
                               midi_2=midi_2, handed_digit_2=handed_digit_2,
                               midi_3=midi_3, handed_digit_3=handed_digit_3)
    return trigram_node


def assess_stretch(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_stretch(trigram=trigram)


def assess_small_span(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_small_span(trigram=trigram)


def assess_large_span(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_large_span(trigram=trigram)


def assess_weak_finger(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_weak_finger(trigram=trigram)


def assess_3_to_4(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_3_to_4(trigram=trigram)


def assess_4_on_black(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_4_on_black(trigram=trigram)


def assess_thumb_on_black(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_thumb_on_black(trigram=trigram)


def assess_5_on_black(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_5_on_black(trigram=trigram)


def assess_thumb_passing(y_prev, y, x_bar, i):
    global judge
    trigram = get_trigram_node(notes=x_bar, i=i, y_prev=y_prev, y=y)
    return judge.assess_thumb_passing(trigram=trigram)


def chord_notes_below(y_prev, y, x_bar, i):
    note_count = x_bar[i]['left_chord']
    return note_count


def chord_notes_above(y_prev, y, x_bar, i):
    note_count = x_bar[i]['right_chord']
    return note_count


def add_functions(xycrf):
    xycrf.add_feature_function(func=assess_stretch, name='str')
    xycrf.add_feature_function(func=assess_small_span, name='sma')
    xycrf.add_feature_function(func=assess_large_span, name='lar')
    xycrf.add_feature_function(func=assess_weak_finger, name='wea')
    xycrf.add_feature_function(func=assess_3_to_4, name='3t4')
    xycrf.add_feature_function(func=assess_4_on_black, name='bl4')
    xycrf.add_feature_function(func=assess_thumb_on_black, name='bl1')
    xycrf.add_feature_function(func=assess_5_on_black, name='bl5')
    xycrf.add_feature_function(func=assess_thumb_passing, name='pa1')
    xycrf.add_feature_function(func=chord_notes_below, name='CNB')
    xycrf.add_feature_function(func=chord_notes_above, name='CNA')


def evaluate_trained_model(the_model, x_test, y_test):
    labels = list(the_model.classes_)
    print(labels)
    y_predicted = my_crf.predict(x_test)
    flat_weighted_f1 = metrics.flat_f1_score(y_test, y_predicted, average='weighted', labels=labels)
    print("Flat weighted F1: {}".format(flat_weighted_f1))

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(y_test, y_predicted, labels=sorted_labels, digits=4))


def train_and_evaluate(the_model, x_train, y_train, x_test, y_test):
    the_model.fit(x_train, y_train)
    evaluate_trained_model(the_model=the_model, x_test=x_test, y_test=y_test)


def get_tag_set(data):
    tag_set = set()
    for (_, y_bar) in data:
        for y in y_bar:
            tag_set.add(y)
    return tag_set


def get_data(x, y):
    data = list()
    for i in range(len(y)):
        example_x: list = copy.deepcopy(x[i])
        example_x.insert(0, {})
        example_x.append({})
        example_y: list = copy.copy(y[i])
        example_y.insert(0, START_TAG)
        example_y.append(STOP_TAG)
        data.append((example_x, example_y))
    return data


def get_dataset_fields(data):
    fields = dict()
    fields['data'] = data
    fields['tag_set'] = get_tag_set(data=data)
    return fields


def split_data(x, y, seed: int, test_size: float, validation_size=0.0):
    splits = {
        'train': dict(),
        'validation': dict(),
        'test': dict()
    }
    data = get_data(x=x, y=y)
    train_size = 1.0 - validation_size - test_size
    (training_data, non_training_data) = train_test_split(data,
                                                          train_size=train_size,
                                                          random_state=seed,
                                                          shuffle=True)
    splits['train'] = get_dataset_fields(data=training_data)

    if validation_size == 0:
        splits['test'] = get_dataset_fields(data=non_training_data)
    elif test_size != 0:
        non_training_size = validation_size + test_size
        scaled_validation_size = validation_size / non_training_size
        (validation_data, test_data) = train_test_split(non_training_data,
                                                        train_size=scaled_validation_size,
                                                        random_state=seed,
                                                        shuffle=True)
        splits['validation'] = get_dataset_fields(data=validation_data)
        splits['test'] = get_dataset_fields(data=test_data)

    return splits


def k_fold_corpus(data, seed: int, k: int, holdout_size=0.15):
    splits = {
        'folds': list(),
        'holdout': dict()
    }
    if holdout_size != 0:
        (data, holdout_data) = train_test_split(data,
                                                test_size=holdout_size,
                                                random_state=seed,
                                                shuffle=True)
        splits['holdout'] = get_dataset_fields(data=holdout_data)
    folder = KFold(n_splits=k, random_state=seed, shuffle=True)
    for train_indices, test_indices in folder.split(data):
        fold_fields = dict()
        fold_train_data = list()
        fold_test_data = list()
        for index in train_indices:
            fold_train_data.append(data[index])
        for index in test_indices:
            fold_test_data.append(data[index])
        fold_fields['train'] = get_dataset_fields(data=fold_train_data)
        fold_fields['test'] = get_dataset_fields(data=fold_test_data)
        splits['folds'].append(fold_fields)
    return splits


#####################################################
# MAIN BLOCK
#####################################################
creal = Corporeal()
judge_model_name = c.VERSION_FEATURES[c.VERSION]['judge']
judge = None
if judge_model_name != 'none':
    judge = creal.get_model(judge_model_name.lower())

corpora_str = "-".join(CORPUS_NAMES)
experiment_name = corpora_str + '__' + TEST_METHOD + '__' + c.VERSION
ex = c.unpickle_it(obj_type="DExperiment", file_name=experiment_name)
if ex is None:
    ex = DExperiment(corpus_names=CORPUS_NAMES, model_version=c.VERSION)
    c.load_data(ex=ex, experiment_name=experiment_name, staffs=STAFFS, corpus_names=CORPUS_NAMES)

ex.print_summary(test_method=TEST_METHOD)

# crf_pickle_file_name = 'crf_' + experiment_name
# have_trained_model = False
# my_crf = c.unpickle_it(obj_type="crf", file_name=crf_pickle_file_name)
# if my_crf:
#     have_trained_model = True
# else:
#     my_crf = XyCrf(optimize=False)


splits = split_data(x=ex.x, y=ex.y, seed=1066, test_size=0.2, validation_size=0.1)
my_crf = XyCrf(optimize=False)
tag_set = splits['train']['tag_set']
my_crf.set_tags(tag_set=tag_set)
add_functions(xycrf=my_crf)
my_crf.training_data = splits['train']['data']
print("Boo")

# if TEST_METHOD == 'cross-validate':
#     scores = cross_val_score(my_crf, ex.x, ex.y, cv=5)
#     # scores = cross_validate(my_crf, ex.x, ex.y, cv=5, scoring="flat_precision_score")
#     print(scores)
#     avg_score = sum(scores) / len(scores)
#     print("Average cross-validation score: {}".format(avg_score))
# elif TEST_METHOD == 'preset':
#     # my_crf.fit(ex.x_train, ex.y_train)
#     if have_trained_model:
#         evaluate_trained_model(the_model=my_crf, x_test=ex.x_test, y_test=ex.y_test)
#     else:
#         train_and_evaluate(the_model=my_crf, x_train=ex.x_train, y_train=ex.y_train, x_test=ex.x_test, y_test=ex.y_test)
#     # total_simple_match_count, total_annot_count, simple_match_rate = ex.get_simple_match_rate(output=True)
#     # result, complex_piece_results = ex.get_complex_match_rates(weight=False)
#     # print("Unweighted avg M for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
#     # result, my_piece_results = ex.get_my_avg_m(weight=False, reuse=False)
#     # print("My unweighted avg m for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
#     # for key in sorted(complex_piece_results):
#         # print("nak {} => {}".format (key, complex_piece_results[key]))
#         # print(" my {} => {}".format(key, my_piece_results[key]))
#         # print("")
#     # result, piece_results = get_complex_match_rates(ex=ex, weight=True)
#     # print("Weighted avg M for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
#     # result, piece_results = get_my_avg_m_gen(ex=ex, weight=True, reuse=True)
#     # print("Weighted avg m_gen for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
# else:
#     split_x_train, split_x_test, split_y_train, split_y_test = \
#         train_test_split(ex.x, ex.y, test_size=0.4, random_state=0)
#     train_and_evaluate(the_model=my_crf, x_train=split_x_train, y_train=split_y_train,
#                        x_test=split_x_test, y_test=split_y_test)
#
# if not have_trained_model:
#     c.pickle_it(obj=my_crf, obj_type='crf', file_name=crf_pickle_file_name)
#
# # unpickled_crf = unpickle_it(obj_type="crf", file_name=pickle_file_name)
# # y_predicted = unpickled_crf.predict(ex.x_test)
# # print("Unpickled CRF result: {}".format(y_predicted))
# # flat_f1 = metrics.flat_f1_score(ex.y_test, y_predicted, average='weighted')
# # print("Unpickled Flat F1: {}".format(flat_f1))
#
# print("Run of crf model {} against {} test set over {} corpus has completed successfully.".format(
#     c.VERSION, TEST_METHOD, corpora_str))
# print("Clean list: {}".format(list(c.CLEAN_LIST.keys())))
