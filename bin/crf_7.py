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
import math
import csv
import statistics
import pprint
import decimal
from statsmodels.stats.weightstats import DescrStatsW
import scipy.stats as stats

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
CORPUS_NAMES = ['complete_layer_one', 'scales', 'arpeggios', 'broken', 'pig']
CORPUS_NAMES = ['scales', 'arpeggios', 'broken', 'pig']
# CORPUS_NAMES = ['pig_training']
# CORPUS_NAMES = ['pig']
# CORPUS_NAMES = ['pig_indy']
# CORPUS_NAMES = ['pig_seg']
# CLEAN_LIST = {'DCorpus': True}
CLEAN_LIST = {}  # Reuse all pickled results.
# CLEAN_LIST = {'crf': True}
# CLEAN_LIST = {'crf': True, 'DExperiment': True}  # Pickles to discard (and regenerate).
# CLEAN_LIST = {'crf': True, 'DCorpus': True, 'DExperiment': True}  # Pickles to discard (and regenerate).
OPTS = {
    'pickling': True,
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
    'test_set': 'pig_test',
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
results = ex.evaluate(the_model=my_crf, is_trained=have_trained_model)
###fold_results = ex.train_and_evaluate_folds(the_model=my_crf, on_train=True, output_results=True)
###ex.summarize_more_fold_results(results=fold_results)

hmm_rows = list()
with open('/Users/dave/tb2/didactyl/dd/corpora/pig/reproduction.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # pprint.pprint(row)
        if row['Pianist_Count']:
            hmm_rows.append(row)

my_report = {'weighted':   {'gen': [], 'high': [], 'soft': [], 'mean': {}, 'std_dev': {}, 'p_val': {}},
             'unweighted': {'gen': [], 'high': [], 'soft': [], 'mean': {}, 'std_dev': {}, 'p_val': {}}}
hmm_report = {'weighted':   {'gen': [], 'high': [], 'soft': [], 'mean': {}, 'std_dev': {}},
              'unweighted': {'gen': [], 'high': [], 'soft': [], 'mean': {}, 'std_dev': {}}}
for method in ('gen', 'high', 'soft'):
    hmm_method = 'M_' + method
    my_seg_results = list()
    for sk in sorted(results['seg_match_rates']):
        my_seg_results.append(results['seg_match_rates'][sk]['combined'][method])
    hmm_seg_results = list()
    note_total = 0
    seg_note_counts = list()
    for row in hmm_rows:
        hmm_seg_results.append(float(row[hmm_method]))
        note_total += int(row['Score_Note_Count'])
        seg_note_counts.append(int(row['Score_Note_Count']))
    my_mean = round(sum(my_seg_results) / len(my_seg_results), 4)
    hmm_mean = round(sum(hmm_seg_results) / len(hmm_seg_results), 4)
    mean_diff = round(my_mean - hmm_mean, 4)
    paired_ttest_result = stats.ttest_rel(my_seg_results, hmm_seg_results)
    p_val = paired_ttest_result.pvalue
    print(f'M_{method} CRF: {my_mean} HMM:{hmm_mean} DIFF: {mean_diff}: {paired_ttest_result}')

    my_weighted_seg_results = list()
    hmm_weighted_seg_results = list()
    seg_weights = list()
    for i in range(len(hmm_rows)):
        seg_weight = seg_note_counts[i] / note_total
        seg_weights.append(seg_weight)
        weighted_val = my_seg_results[i] * seg_weight
        my_weighted_seg_results.append(weighted_val)
        weighted_val = hmm_seg_results[i] * seg_weight
        hmm_weighted_seg_results.append(weighted_val)
    my_weighted_mean = round(sum(my_weighted_seg_results), 4)
    hmm_weighted_mean = round(sum(hmm_weighted_seg_results), 4)
    weighted_mean_diff = round(my_weighted_mean - hmm_weighted_mean, 4)
    paired_ttest_result = stats.ttest_rel(my_weighted_seg_results, hmm_weighted_seg_results)
    weighted_p_val = paired_ttest_result.pvalue
    print(f'Weighted M_{method} CRF: {my_weighted_mean} HMM:{hmm_weighted_mean} DIFF: {weighted_mean_diff}: {paired_ttest_result}')
    my_report['unweighted'][method] = my_seg_results
    my_report['weighted'][method] = my_weighted_seg_results
    hmm_report['unweighted'][method] = hmm_seg_results
    hmm_report['weighted'][method] = hmm_weighted_seg_results
    my_report['unweighted']['mean'][method] = my_mean
    my_report['weighted']['mean'][method] = my_weighted_mean
    hmm_report['unweighted']['mean'][method] = hmm_mean
    hmm_report['weighted']['mean'][method] = hmm_weighted_mean
    my_report['unweighted']['std_dev'][method] = results['std_devs']['combined'][method]
    my_report['weighted']['std_dev'][method] = results['weighted_std_devs']['combined'][method]
    hmm_report['unweighted']['std_dev'][method] = statistics.stdev(hmm_seg_results)
    hmm_report['weighted']['std_dev'][method] = DescrStatsW(hmm_seg_results, weights=seg_weights).std
    my_report['unweighted']['p_val'][method] = p_val
    my_report['weighted']['p_val'][method] = weighted_p_val

# pprint.pprint(my_report)

# print("\nWeight & Measure & HMM_mean & CRF_mean & SD_HMM & SD_CRF & Mean_Diff & p_value")
print("\nWeight & Measure & HMM_mean & CRF_mean & Mean_Diff & p_value")
for weight in ('weighted', 'unweighted'):
    for method in ('gen', 'high', 'soft'):
        latex_method = '$M_{' + method + '}$'
        my_mean = '{:0.4f}'.format(my_report[weight]['mean'][method])
        my_sd = '{:0.4f}'.format(my_report[weight]['std_dev'][method])
        hmm_mean = '{:0.4f}'.format(hmm_report[weight]['mean'][method])
        hmm_sd = '{:0.4f}'.format(hmm_report[weight]['std_dev'][method])
        mean_diff = '{:0.4f}'.format(float(my_mean) - float(hmm_mean))
        p_val = '{:0.3f}'.format(my_report[weight]['p_val'][method])
        # row_str = f"{weight} & {latex_method} & {my_mean} & {hmm_mean} & {my_sd} & {hmm_sd} & {mean_diff} & {p_val} \\\\"
        row_str = f"{weight} & {latex_method} & {my_mean} & {hmm_mean} & {mean_diff} & {p_val} \\\\"
        print(row_str)

print("\nSegment & Notes & Pianists & CRF $M_{gen}$ & HMM & CRF $M_{high}$ & HMM & CRF $M_{soft}$ & HMM")
for weighted in ('weighted', 'unweighted'):
    row_index = 0
    print(f'{weighted} RESULTS:')
    for row in hmm_rows:
        row_str = f"{row['Score_ID']} & {row['Score_Note_Count']} & {row['Pianist_Count']}"
        for method in ('gen', 'high', 'soft'):
            my_val = my_report[weighted][method][row_index]
            hmm_val = hmm_report[weighted][method][row_index]
            my_val_str = '{:.5f}'.format(round(my_val, 5))
            hmm_val_str = '{:.5f}'.format(round(hmm_val, 5))
            if my_val > hmm_val:
                my_val_str = "\\textbf{" + my_val_str + '}'
            else:
                hmm_val_str = "\\textbf{" + hmm_val_str + '}'
            row_str += ' & ' + my_val_str + ' & ' + hmm_val_str

        print(row_str + " \\\\")
        row_index += 1

if not have_trained_model and OPTS['pickling']:
    c.pickle_it(obj=my_crf, obj_type="crf", file_name=experiment_name, use_dill=True)
