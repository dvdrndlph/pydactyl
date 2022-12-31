#!/usr/bin/env python
__author__ = 'David Randolph'
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
from sklearn.model_selection import train_test_split, cross_val_score
from pydactyl.util.DExperiment import DExperiment
from music21 import note
import pydactyl.util.CrfUtil as c


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
# CORPUS_NAMES = ['scales', 'arpeggios', 'broken']
# CORPUS_NAMES = ['pig']
# CORPUS_NAMES = ['pig_indy']
CORPUS_NAMES = ['pig_seg']


#####################################################
# FUNCTIONS
#####################################################


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


def my_note2features(notes, i, staff, categorical=False):
    features = {}

    features['BOP'] = "0"
    if i == 0:
        features['BOP'] = "1"
    features['EOP'] = "0"
    if i >= len(notes) - 1:
        features['EOP'] = "1"

    x_d = dict()
    y_d = dict()
    x_d[-4], y_d[-4] = c.lattice_distance(notes=notes, from_i=i-4, to_i=i)
    x_d[-3], y_d[-3] = c.lattice_distance(notes=notes, from_i=i-3, to_i=i)
    x_d[-2], y_d[-2] = c.lattice_distance(notes=notes, from_i=i-2, to_i=i)
    x_d[-1], y_d[-1] = c.lattice_distance(notes=notes, from_i=i-1, to_i=i)
    x_d[+1], y_d[+1] = c.lattice_distance(notes=notes, from_i=i, to_i=i+1)
    x_d[+2], y_d[+2] = c.lattice_distance(notes=notes, from_i=i, to_i=i+2)
    x_d[+3], y_d[+3] = c.lattice_distance(notes=notes, from_i=i, to_i=i+3)
    x_d[+4], y_d[+4] = c.lattice_distance(notes=notes, from_i=i, to_i=i+4)

    features['x_distance:-4'], features['y_distance:-4'] = x_d[-4], y_d[-4]
    features['x_distance:-3'], features['y_distance:-3'] = x_d[-3], y_d[-3]
    features['x_distance:-2'], features['y_distance:-2'] = x_d[-2], y_d[-2]
    features['x_distance:-1'], features['y_distance:-1'] = x_d[-1], y_d[-1]
    features['x_distance:+1'], features['y_distance:+1'] = x_d[+1], y_d[+1]
    features['x_distance:+2'], features['y_distance:+2'] = x_d[+2], y_d[+2]
    features['x_distance:+3'], features['y_distance:+3'] = x_d[+3], y_d[+3]
    features['x_distance:+4'], features['y_distance:+4'] = x_d[+4], y_d[+4]

    # features['y_gram'] = "{}|{}|{}|{}".format(y_d[-2], y_d[-1], y_d[1], y_d[2])

    # Chord features. Approximate with 30 ms offset deltas a la Nakamura.
    left_chord_notes, right_chord_notes = c.chordings(notes=notes, middle_i=i)
    features['left_chord'] = left_chord_notes
    features['right_chord'] = right_chord_notes

    features['staff'] = 0
    if staff == "upper":
        features['staff'] = 1
        # @100: [0.54495717 0.81059147 0.81998371 0.68739401 0.73993751]
        # @1:   [0.54408935 0.80563961 0.82079826 0.6941775  0.73534277]

    black = dict()
    black[0] = str(c.black_key(notes, i))
    black[-1] = str(c.black_key(notes, i-1))
    black[1] = str(c.black_key(notes, i+1))
    # features['blackgram'] = "{}|{}".format(black[-1], black[0])

    # 56.83
    features['black:-1'] = black[-1]
    features['black'] = black[0]
    features['black:+1'] = black[1]

    # 57.18
    features['level_change'] = 0
    if y_d[-1] != 0:
        features['level_change'] = 1

    features['returning'] = 0
    if x_d[-2] == 0:
        features['returning'] = 1  # .5486
    features['will_return'] = 0
    if x_d[+2] == 0:
        features['will_return'] = 1  # .5562

    # 57.18 w/both
    features['ascending'] = 0
    if x_d[-1] < 0 and x_d[+1] > 0:
        features['ascending'] = 1
    features['descending'] = 0
    if x_d[-1] > 0 and x_d[+1] < 0:
        features['descending'] = 1

    # pit = dict()
    # pit[-3], pit[-2], pit[-1], pit[0], pit[1], pit[2], pit[3] = c.get_pit_strings(notes, i, range=3)
    #
    # features['pit_-1|0'] = pit[-1] + '|' + pit[0]
    # features['pit_0|+1'] = pit[0] + '|' + pit[1]
    # features['pit_-1|0|+1'] = pit[-1] + '|' + pit[0] + pit[1]

    # Impact of large leaps? Costs max out, no? Maybe not.
    features['leap'] = 0
    if c.leap_is_excessive(notes, i):
        features['leap'] = 1

    oon = notes[i]
    m21_note: note.Note = oon['note']
    on_velocity = m21_note.volume.velocity
    if on_velocity is None:
        on_velocity = 64
    features['velocity'] = on_velocity

    tempi = c.tempo_features(notes=notes, middle_i=i)
    for k in tempi:
        features[k] = tempi[k]

    # arts = c.articulation_features(notes=notes, middle_i=i)
    # for k in arts:
    #     features[k] = arts[k]

    # reps_before, reps_after = c.repeat_features(notes=notes, middle_i=i)
    # features['repeats_before'] = reps_before
    # features['repeats_after'] = reps_after

    return features


#####################################################
# MAIN BLOCK
#####################################################
corpora_str = "-".join(CORPUS_NAMES)
experiment_name = corpora_str + '__' + TEST_METHOD + '__3'
ex = c.unpickle_it(obj_type="DExperiment", file_name=experiment_name)
if ex is None:
    ex = DExperiment(corpus_names=CORPUS_NAMES, model_version="3", note_func=my_note2features)
    c.load_data(ex=ex, experiment_name=experiment_name, staffs=STAFFS, corpus_names=CORPUS_NAMES)

ex.print_summary(test_method=TEST_METHOD)

crf_pickle_file_name = 'crf_' + experiment_name
have_trained_model = False
my_crf = c.unpickle_it(obj_type="crf", file_name=crf_pickle_file_name)
if my_crf:
    have_trained_model = True
else:
    my_crf = crf.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        # max_iterations=100,
        all_possible_transitions=True
    )

if TEST_METHOD == 'cross-validate':
    scores = cross_val_score(my_crf, ex.x, ex.y, cv=5)
    # scores = cross_validate(my_crf, ex.x, ex.y, cv=5, scoring="flat_precision_score")
    print(scores)
    avg_score = sum(scores) / len(scores)
    print("Average cross-validation score: {}".format(avg_score))
elif TEST_METHOD == 'preset':
    # my_crf.fit(ex.x_train, ex.y_train)
    if have_trained_model:
        evaluate_trained_model(the_model=my_crf, x_test=ex.x_test, y_test=ex.y_test)
    else:
        train_and_evaluate(the_model=my_crf, x_train=ex.x_train, y_train=ex.y_train, x_test=ex.x_test, y_test=ex.y_test)
    # total_simple_match_count, total_annot_count, simple_match_rate = ex.get_simple_match_rate(output=True)
    # result, complex_piece_results = ex.get_complex_match_rates(weight=False)
    # print("Unweighted avg M for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
    # result, my_piece_results = ex.get_my_avg_m(weight=False, reuse=False)
    # print("My unweighted avg m for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
    # for key in sorted(complex_piece_results):
        # print("nak {} => {}".format (key, complex_piece_results[key]))
        # print(" my {} => {}".format(key, my_piece_results[key]))
        # print("")
    # result, piece_results = get_complex_match_rates(ex=ex, weight=True)
    # print("Weighted avg M for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
    # result, piece_results = get_my_avg_m_gen(ex=ex, weight=True, reuse=True)
    # print("Weighted avg m_gen for crf{} over {}: {}".format(VERSION, CORPUS_NAMES, result))
else:
    split_x_train, split_x_test, split_y_train, split_y_test = \
        train_test_split(ex.x, ex.y, test_size=0.4, random_state=0)
    train_and_evaluate(the_model=my_crf, x_train=split_x_train, y_train=split_y_train,
                       x_test=split_x_test, y_test=split_y_test)

if not have_trained_model:
    c.pickle_it(obj=my_crf, obj_type='crf', file_name=crf_pickle_file_name)

# unpickled_crf = unpickle_it(obj_type="crf", file_name=pickle_file_name)
# y_predicted = unpickled_crf.predict(ex.x_test)
# print("Unpickled CRF result: {}".format(y_predicted))
# flat_f1 = metrics.flat_f1_score(ex.y_test, y_predicted, average='weighted')
# print("Unpickled Flat F1: {}".format(flat_f1))

print("Run of crf model {} against {} test set over {} corpus has completed successfully.".format(
    "3", TEST_METHOD, corpora_str))
print("Clean list: {}".format(list(c.CLEAN_LIST.keys())))
