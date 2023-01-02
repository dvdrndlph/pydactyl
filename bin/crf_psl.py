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
import time
import pprint
from datetime import datetime
from pathlib import Path

from pyseqlab.utilities import SequenceStruct
from pyseqlab.attributes_extraction import GenericAttributeExtractor
from pyseqlab.utilities import TemplateGenerator
from pyseqlab.features_extraction import FeatureExtractor, FOFeatureExtractor
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.workflow import GenericTrainingWorkflow
from pyseqlab.ho_crf_ad import HOCRFAD, HOCRFADModelRepresentation
from sklearn_crfsuite import metrics
from pydactyl.util.DExperiment import DExperiment
import pydactyl.crf.Crf3 as crf3
import pydactyl.util.CrfUtil as c

# CROSS_VALIDATE = False
# One of 'cross-validate', 'preset', 'random'
# TEST_METHOD = 'cross-validate'
# TEST_METHOD = 'preset'
TEST_METHOD = 'random'
STAFFS = ['upper', 'lower']
# STAFFS = ['upper']
# STAFFS = ['lower']

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


def get_sequence_struct_list(X, Y):
    seqs = []
    example_index = 0
    for phrase in X:
        fingers = Y[example_index]
        seq = SequenceStruct(X=phrase, Y=fingers)
        seqs.append(seq)
        example_index += 1
    return seqs


def get_attr_desc_dict(X):
    ad_dict = {}
    # 'BOP': {'description': 'beginning of phrase', 'encoding': 'categorical'},
    # 'EOP': {'description': 'end of phrase', 'encoding': 'categorical'},
    attr_names = X[0][1].keys()
    for name in attr_names:
        if type(X[0][1][name]) is str:
            ad_dict[name] = {'encoding': 'categorical'}
        else:
            ad_dict[name] = {'encoding': 'continuous'}
    # What's this about integers being categorical in filters?!
    return ad_dict


def generate_templates_alltracks(attr_desc):
    tg = TemplateGenerator()
    template_XY = dict()
    for track_attr_name in attr_desc:
        tg.generate_template_XY(track_attr_name, ('1-gram', range(0, 1)), '1-state:2-state', template_XY)
    template_Y = tg.generate_template_Y('1-state')
    return template_XY, template_Y


def generate_attributes_alltracks(attr_extractor, seq):
    print("my_attr_extractor.attr_desc")
    print(attr_extractor.attr_desc)
    print("-" * 40)
    print("sequence")
    print(seq)
    seq.seg_attr.clear()
    attr_extractor.generate_attributes(seq, seq.get_y_boundaries())
    print("extracted attributes saved in seq.seg_attr")
    print(seq.seg_attr)
    print("-"*40)


def get_workflow(ex, working_dir, seqs):
    attr_desc = get_attr_desc_dict(X=ex.x)
    ax = GenericAttributeExtractor(attr_desc)
    seq_1 = seqs[0]
    template_XY, template_Y = generate_templates_alltracks(attr_desc)
    generate_attributes_alltracks(attr_extractor=ax, seq=seq_1)

    fx = FOFeatureExtractor(templateX=template_XY, templateY=template_Y, attr_desc=attr_desc)
    extracted_features = fx.extract_seq_features_perboundary(seq_1)
    print("extracted features")
    print(extracted_features)

    fx_filter = None
    workflow = GenericTrainingWorkflow(ax, fx, fx_filter,
                                       FirstOrderCRFModelRepresentation, FirstOrderCRF,
                                       working_dir)
    return workflow


def create_focrf_model(ex, working_dir):
    train_seqs = get_sequence_struct_list(X=ex.x, Y=ex.y)
    # train_seqs = get_sequence_struct_list(X=ex.x_train, Y=ex.y_train)
    # test_seqs = get_sequence_struct_list(X=ex.x_test, Y=ex.y_test)
    workflow = get_workflow(ex, working_dir, seqs=train_seqs)
    # use all passed data as training data -- no splitting
    data_split_options = {'method': 'none'}
    data_split = workflow.seq_parsing_workflow(data_split_options, seqs=train_seqs, full_parsing=True)
    print()
    print("data_split: 'none' option ")
    print(data_split)
    print()
    # folder name will be f_0 as fold 0
    crf_m = workflow.build_crf_model(data_split[0]['train'], "f_0")
    print()
    print("type of built model:")
    print(type(crf_m))
    print()
    print("number of generated features:")
    print(len(crf_m.model.modelfeatures_codebook))
    # print("features:")
    # pprint.pprint(crf_m.model.modelfeatures)

    # use L-BFGS-B method for training
    optimization_options = {
        "method": "L-BFGS-B",
        "regularization_type": "l2",
        "regularization_value": 0,
        "maxiter": 2
    }
    # use SGA method for training
    optimization_options = {
        "method": "SGA",
        "num_epochs": 5
    }
    # start with 0 weights
    crf_m.weights.fill(0)
    train_seqs_id = data_split[0]['train']
    model_dir = workflow.train_model(train_seqs_id, crf_m, optimization_options)
    print("Trained model directory: {}".format(model_dir))
    print("*" * 50)


def evaluate_trained_model(ex, working_dir, model_dir):
    test_seqs = get_sequence_struct_list(X=ex.x, Y=ex.y)
    # test_seqs = get_sequence_struct_list(X=ex.x_test, Y=ex.y_test)
    workflow = get_workflow(ex, working_dir, seqs=test_seqs)
    data_split_options = {'method': 'none'}
    data_split = workflow.seq_parsing_workflow(data_split_options, seqs=test_seqs, full_parsing=True)

    opts = {
        'seqs_info': workflow.seqs_info,
        'model_eval': True,
        'metric': 'f1'
    }
    perf = workflow.use_model(model_dir, opts)
    metric = opts['metric']
    print("metric {}: {}".format(metric, perf[metric]))

#####################################################
# MAIN BLOCK
#####################################################
version_str = 'psl_3'
start_dt = datetime.now()
corpora_str = "-".join(CORPUS_NAMES)
staff_str = "-".join(STAFFS)
WORKING_DIR = '/users/dave/pyseqlab/' + corpora_str + '/' + staff_str + '/' + version_str
model_dir = '/users/dave/pyseqlab/scales-arpeggios-broken/upper-lower/psl_3/working_dir/models/2023_1_1-23_25_9_763552'
model_dir = '/users/dave/pyseqlab/scales-arpeggios-broken/upper-lower/psl_3/working_dir/models/2023_1_2-0_54_55_105891'
experiment_name = corpora_str + '__' + TEST_METHOD + '__' + version_str
ex = c.unpickle_it(obj_type="DExperiment", file_name=experiment_name)
if ex is None:
    ex = DExperiment(corpus_names=CORPUS_NAMES, model_version=version_str, note_func=crf3.my_note2features)
    c.load_data(ex=ex, experiment_name=experiment_name, staffs=STAFFS, corpus_names=CORPUS_NAMES)

ex.print_summary(test_method=TEST_METHOD)
evaluate_trained_model(ex=ex, working_dir=WORKING_DIR, model_dir=model_dir)
exit(0)
create_focrf_model(ex=ex, working_dir=WORKING_DIR)
exit(0)

#
# # working_path = Path(WORKING_DIR)
# # have_model = False
# # if working_path.is_dir():
#     # have_model = True
# # if not have_model:

end_dt = datetime.now()
execution_duration_minutes = (end_dt - start_dt)
print("Total running time (wall clock): {}".format(execution_duration_minutes))
exit(0)

if TEST_METHOD == 'cross-validate':
    # scores = cross_val_score(my_crf, ex.x, ex.y, cv=5)
    # scores = cross_validate(my_crf, ex.x, ex.y, cv=5, scoring="flat_precision_score")
    scores = []
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
# else:
#     split_x_train, split_x_test, split_y_train, split_y_test = \
#         train_test_split(ex.x, ex.y, test_size=0.4, random_state=0)
#     train_and_evaluate(the_model=my_crf, x_train=split_x_train, y_train=split_y_train,
#                        x_test=split_x_test, y_test=split_y_test)

if not have_trained_model:
    c.pickle_it(obj=my_crf, obj_type='crf', file_name=crf_pickle_file_name)

# unpickled_crf = unpickle_it(obj_type="crf", file_name=pickle_file_name)
# y_predicted = unpickled_crf.predict(ex.x_test)
# print("Unpickled CRF result: {}".format(y_predicted))
# flat_f1 = metrics.flat_f1_score(ex.y_test, y_predicted, average='weighted')
# print("Unpickled Flat F1: {}".format(flat_f1))

print("Run of crf model {} against {} test set over {} corpus has completed successfully.".format(
    c.VERSION, TEST_METHOD, corpora_str))
print("Clean list: {}".format(list(c.CLEAN_LIST.keys())))
