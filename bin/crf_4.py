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
import sklearn_crfsuite as crf
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from pydactyl.util.DExperiment import DExperiment
import pydactyl.util.CrfUtil as c
import pydactyl.crf.CrfFeatures4 as model;

# CROSS_VALIDATE = False
# One of 'cross-validate', 'preset', 'random'
# TEST_METHOD = 'cross-validate'
TEST_METHOD = 'preset'
# TEST_METHOD = 'random'
STAFFS = ['upper', 'lower']
# STAFFS = ['upper']
# STAFFS = ['lower']
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
    return y_predicted


def train_and_evaluate(the_model, x_train, y_train, x_test, y_test):
    the_model.fit(x_train, y_train)
    return evaluate_trained_model(the_model=the_model, x_test=x_test, y_test=y_test)


#####################################################
# MAIN BLOCK
#####################################################
corpora_str = "-".join(CORPUS_NAMES)
experiment_name = corpora_str + '__' + TEST_METHOD + '__' + model.CRF_VERSION
ex = c.unpickle_it(obj_type="DExperiment", file_name=experiment_name)
if ex is None:
    ex = DExperiment(corpus_names=CORPUS_NAMES, model_version=model.CRF_VERSION,
                     note_func=model.my_note2features, reverse=model.REVERSE_NOTES)
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
    if have_trained_model:
        predictions = evaluate_trained_model(the_model=my_crf, x_test=ex.x_test, y_test=ex.y_test)
    else:
        predictions = train_and_evaluate(the_model=my_crf, x_train=ex.x_train,
                                         y_train=ex.y_train, x_test=ex.x_test, y_test=ex.y_test)
    total_simple_match_count, total_annot_count, simple_match_rate = \
        ex.get_simple_match_rate(predictions=predictions, output=True)
    print("Simple match rate: {}".format(simple_match_rate))
    # result, complex_piece_results = ex.get_complex_match_rates(predictions=predictions, weight=False)
    # print("Unweighted avg M for crf{} over {}: {}".format(model.CRF_VERSION, CORPUS_NAMES, result))
    result, my_piece_results = ex.get_my_avg_m(predictions=predictions, weight=False, reuse=False)
    print("My unweighted avg M for crf{} over {}: {}".format(model.CRF_VERSION, CORPUS_NAMES, result))
    # for key in sorted(complex_piece_results):
        # print("nak {} => {}".format (key, complex_piece_results[key]))
        # print(" my {} => {}".format(key, my_piece_results[key]))
        # print("")
    # result, piece_results = ex.get_complex_match_rates(weight=True)
    # print("Weighted avg M for crf{} over {}: {}".format(model.CRF_VERSION, CORPUS_NAMES, result))
    result, piece_results = ex.get_my_avg_m(predictions=predictions, weight=True, reuse=True)
    print("Weighted avg m_gen for crf{} over {}: {}".format(model.CRF_VERSION, CORPUS_NAMES, result))
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
    model.CRF_VERSION, TEST_METHOD, corpora_str))
print("Clean list: {}".format(list(c.CLEAN_LIST.keys())))
