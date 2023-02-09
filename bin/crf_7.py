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
from pydactyl.eval.DExperiment import DExperiment, DExperimentOpts
import pydactyl.crf.CrfUtil as c
import pydactyl.crf.CrfFeatures7 as feats

# One of 'cross-validate', 'preset', 'random'
# TEST_METHOD = 'cross-validate'
# TEST_METHOD = 'preset'
# TEST_METHOD = 'random'
# STAFFS = ['upper', 'lower']
# STAFFS = ['upper']
# STAFFS = ['lower']
# CORPUS_NAMES = ['full_american_by_annotator']
# CORPUS_NAMES = ['complete_layer_one']
# CORPUS_NAMES = ['scales']
# CORPUS_NAMES = ['arpeggios']
# CORPUS_NAMES = ['broken']
# CORPUS_NAMES = ['scales', 'arpeggios', 'broken']
# CORPUS_NAMES = ['complete_layer_one', 'scales', 'arpeggios', 'broken']
CORPUS_NAMES = ['pig_training']
CORPUS_NAMES = ['pig']
# CORPUS_NAMES = ['pig_indy']
# CORPUS_NAMES = ['pig_seg']
# CLEAN_LIST = {'DCorpus': True}
CLEAN_LIST = {}  # Reuse all pickled results.
# CLEAN_LIST = {'crf': True}
CLEAN_LIST = {'crf': True, 'DExperiment': True}  # Pickles to discard (and regenerate).
CLEAN_LIST = {'crf': True, 'DCorpus': True, 'DExperiment': True}  # Pickles to discard (and regenerate).
OPTS = {
    'pickling': False,
    'segmenting': False,
    'consonance_threshold': c.CHORD_MS_THRESHOLD,
    'engine': 'sklearn-crfsuite',
    'model_features': feats,
    'corpus_names': CORPUS_NAMES,
    'randomize_corpora': False,  # Must set to False for preset evaluation.
    'staffs': ['upper', 'lower'],
    'test_method': 'preset',  # FIXME: What is this really doing? It does the preset eval up front if possible, no?
                              # Otherwise is evaluates a random split upfront. Is this setting really needed?
    'fold_count': 5,
    'group_by': 'score',  # score, segment, or example (none)
    'holdout_predefined': True,
    'holdout_size': 0.3,
    'segregate_hands': False,
    'param_grid': {
        'c1': [0, 0.0001, 0.001, 0.01, 0.1],
        'c2': [1.0, 0.5, 0.25, 0.125, 0.1],
        'linesearch': ['MoreThuente', 'Backtracking', 'StrongBacktracking']
    },
    'params': {
        'algorithm': 'lbfgs',
        'c1': 0,  # 0.1,
        'c2': 1.0,  # 0.1,
        'epsilon': 0.00001,
        'period': 10,
        'delta': 0.00001,
        'linesearch': 'MoreThuente',
        'max_linesearch': 20,
        'max_iterations': None,
        'all_possible_transitions': False  # True
    }
}
opts = DExperimentOpts(opts=OPTS)

#####################################################
# MAIN BLOCK
#####################################################
ex = c.unpickle_it(obj_type="DExperiment", clean_list=CLEAN_LIST, opts=opts, use_dill=True)
if ex is None:
    ex = DExperiment(opts=opts)
    experiment_name = ex.load_data(clean_list=CLEAN_LIST)
    if OPTS['pickling']:
        c.pickle_it(obj=ex, obj_type="DExperiment", file_name=experiment_name, use_dill=True)
ex.print_summary()

experiment_name = ex.experiment_name()
have_trained_model = False
my_crf = c.unpickle_it(obj_type="crf", clean_list=CLEAN_LIST, opts=opts, use_dill=True)
if my_crf:
    have_trained_model = True
else:
    my_crf = crf.CRF(
        algorithm=OPTS['params']['algorithm'],
        c1=0.1,
        c2=0.1,
        # max_iterations=100,
        all_possible_transitions=True
    )

# ex.tune_parameters(the_model=my_crf)
ex.evaluate(the_model=my_crf, is_trained=have_trained_model)
fold_results = ex.train_and_evaluate_folds(the_model=my_crf, on_train=True, output_results=True)
ex.summarize_more_fold_results(results=fold_results)
if not have_trained_model and OPTS['pickling']:
    c.pickle_it(obj=my_crf, obj_type="crf", file_name=experiment_name, use_dill=True)
