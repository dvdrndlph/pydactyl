#!/usr/bin/env python
__author__ = 'David Randolph'

import os
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
from pyseqlab.utilities import generate_trained_model
from pyseqlab.utilities import SequenceStruct
from pyseqlab.attributes_extraction import GenericAttributeExtractor
from pyseqlab.utilities import TemplateGenerator
from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.workflow import GenericTrainingWorkflow
from pyseqlab.ho_crf_ad import HOCRFAD, HOCRFADModelRepresentation
from sklearn_crfsuite import metrics
from pydactyl.util.DExperiment import DExperiment
import pydactyl.crf.CrfFeatures3 as crf3
import pydactyl.util.CrfUtil as c

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
        tg.generate_template_XY(track_attr_name, ('1-gram', range(0, 1)), '1-state:2-states:3-states', template_XY)
    template_Y = tg.generate_template_Y('1-state:2-states:3-states')
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


def get_workflow(ex, working_dir, seqs, high_order=True):
    attr_desc = get_attr_desc_dict(X=ex.x)
    ax = GenericAttributeExtractor(attr_desc)
    seq_1 = seqs[0]
    template_XY, template_Y = generate_templates_alltracks(attr_desc)
    generate_attributes_alltracks(attr_extractor=ax, seq=seq_1)

    if high_order:
        fx = HOFeatureExtractor(templateX=template_XY, templateY=template_Y, attr_desc=attr_desc)
        extracted_features = fx.extract_seq_features_perboundary(seq_1)
        print("extracted features")
        print(extracted_features)

        fx_filter = None
        workflow = GenericTrainingWorkflow(ax, fx, fx_filter, HOCRFADModelRepresentation, HOCRFAD, working_dir)
    else:
        fx = FOFeatureExtractor(templateX=template_XY, templateY=template_Y, attr_desc=attr_desc)
        extracted_features = fx.extract_seq_features_perboundary(seq_1)
        print("extracted features")
        print(extracted_features)

        fx_filter = None
        workflow = GenericTrainingWorkflow(ax, fx, fx_filter,
                                           FirstOrderCRFModelRepresentation, FirstOrderCRF,
                                           working_dir)
    return workflow


def create_crf_model(ex, working_dir, optimization_options, weights=None, high_order=True):
    train_seqs = get_sequence_struct_list(X=ex.x_train, Y=ex.y_train)
    # test_seqs = get_sequence_struct_list(X=ex.x_test, Y=ex.y_test)
    workflow = get_workflow(ex, working_dir, seqs=train_seqs, high_order=high_order)
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

    if weights is not None:
        raise Exception("Cannot initialize weights yet. Seems like we might want to sometime.")
    else:
        # start with 0 weights
        crf_m.weights.fill(0)
    train_seqs_id = data_split[0]['train']
    my_model_dir = workflow.train_model(train_seqs_id, crf_m, optimization_options)
    print("Trained model directory: {}".format(my_model_dir))
    print("*" * 50)
    return my_model_dir

def evaluate_trained_model(ex, experiment_name, model_dir):
    attr_desc = get_attr_desc_dict(X=ex.x_train)
    ax = GenericAttributeExtractor(attr_desc)
    model_parts_dir = os.path.join(model_dir, 'model_parts')
    model = generate_trained_model(model_parts_dir, ax)
    decoding_method = 'viterbi'
    output_dir = '/tmp/experiment_name/' + experiment_name
    sep = '\t'

    # test_seqs = get_sequence_struct_list(X=ex.x, Y=ex.y)
    test_seqs = get_sequence_struct_list(X=ex.x_test, Y=ex.y_test)
    # decoded_sequences = model.decode_seqs(decoding_method, output_dir, seqs=test_seqs,
    #                                       file_name='decoding.txt', sep=sep)
    decoded_seqs = model.decode_seqs(decoding_method, output_dir, seqs=test_seqs, sep=sep)
    print()
    y_predicted = list()
    y_test = list()
    labels = set()
    for seq_id in sorted(decoded_seqs):
        print("seq_id ", seq_id)
        print("predicted labels:")
        guess = decoded_seqs[seq_id]['Y_pred']
        for label in guess:
            labels.add(label)
        print(guess)
        y_predicted.append(guess)
        print("reference labels:")
        truth = decoded_seqs[seq_id]['seq'].flat_y
        print(truth)
        y_test.append(truth)
        print("-" * 50)
    labels = list(labels)
    flat_weighted_f1 = metrics.flat_f1_score(y_test, y_predicted, average='weighted', labels=labels)
    print("Flat weighted F1: {}".format(flat_weighted_f1))

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(y_test, y_predicted, labels=sorted_labels, digits=4))

    # total_simple_match_count, total_annot_count, simple_match_rate = \
        # ex.get_simple_match_rate_for_predictions(predictions=y_predicted, output=True)
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


#####################################################
# MAIN BLOCK
#####################################################
version_str = 'psl_3'
start_dt = datetime.now()
corpora_str = "-".join(CORPUS_NAMES)
staff_str = "-".join(STAFFS)
WORKING_DIR = os.path.expanduser('~/pyseqlab/' + corpora_str + '/' + staff_str + '/' + version_str)
model_dir = WORKING_DIR + '/working_dir/models/2023_1_2-2_28_33_167790'
model_dir = WORKING_DIR + '/working_dir/models/2023_1_5-2_49_3_151841'
model_dir = WORKING_DIR + '/working_dir/models/2023_1_9-2_49_34_776109'  # Crf3, lbfgs 100 iterations
model_dir = WORKING_DIR + '/working_dir/models/2023_1_9-11_2_19_643426'  # Crf3, 1-state Y, 2-states XY, lbfgs 200 iterations
model_dir = WORKING_DIR + '/working_dir/models/2023_1_9-21_46_28_197387'  # Crf3, 3-states (Y and XY) lbfgs 200 iterations
experiment_name = corpora_str + '__' + TEST_METHOD + '__' + version_str
ex = c.unpickle_it(obj_type="DExperiment", file_name=experiment_name)
if ex is None:
    ex = DExperiment(corpus_names=CORPUS_NAMES, model_version=version_str, note_func=crf3.my_note2features)
    c.load_data(ex=ex, experiment_name=experiment_name, staffs=STAFFS, corpus_names=CORPUS_NAMES)

ex.print_summary(test_method=TEST_METHOD)
evaluate_trained_model(ex=ex, experiment_name=experiment_name, model_dir=model_dir)
exit(0)

# use L-BFGS-B method for training
optimization_options = {
    "method": "L-BFGS-B",
    "regularization_type": "l2",
    "regularization_value": 0,
    "maxiter": 200
}
# use SGA method for training
# optimization_options = {
#     "method": "SGA",
#     "num_epochs": 5
# }
my_model_dir = create_crf_model(ex=ex, optimization_options=optimization_options, working_dir=WORKING_DIR)
end_dt = datetime.now()
execution_duration_minutes = (end_dt - start_dt)
print("Total running time (wall clock): {}".format(execution_duration_minutes))
# FIXME: Print details about model generation to model directory.
exit(0)
