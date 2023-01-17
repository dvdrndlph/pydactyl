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
# First-order linear chain CRF piano fingering models using sklearn-crfsuite,
# which does not seem to provide a way to predefine "edge-observation" functions
# over both observations and labels.
#
# We retain this script because it slightly outperforms the results in crf_2,
# even though the model logic should be identical. This seems to indicate
# the sklearn_crfsuite is behaving in a non-deterministic way. I swear
# I have witnessed this before, so I want to preserve code to demonstrate the issue.
# The main difference is in moving DExperiment to a different file in crf_2.
import copy
import shutil
import os
import sklearn_crfsuite as crf
from sklearn_crfsuite import metrics
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from music21 import note
from pydactyl.eval.Corporeal import Corporeal, ARPEGGIOS_STD_PIG_DIR, SCALES_STD_PIG_DIR, BROKEN_STD_PIG_DIR, COMPLETE_LAYER_ONE_STD_PIG_DIR
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
from pydactyl.dcorpus.DAnnotation import DAnnotation
from pydactyl.dcorpus.ABCDHeader import ABCDHeader
from pydactyl.dcorpus.PigInOut import PigIn, PigOut, PIG_STD_DIR, PIG_FILE_SUFFIX, PIG_SEGREGATED_STD_DIR
import pydactyl.crf.CrfUtil as c

VERSION = '0003'
PREDICTION_DIR = '/tmp/crf' + VERSION + 'prediction/'
TEST_DIR = '/tmp/crf' + VERSION + 'test/'
PICKLE_BASE_DIR = '/tmp/pickle/'
MAX_LEAP = 16
CHORD_MS_THRESHOLD = 30

# CROSS_VALIDATE = False
# One of 'cross-validate', 'preset', 'random'
# TEST_METHOD = 'cross-validate'
TEST_METHOD = 'preset'
# TEST_METHOD = 'random'
SEGREGATE_HANDS = False
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


def note2features(notes, i, staff):
    settings = c.VERSION_FEATURES[VERSION]
    features = {}

    if settings['bop'] and i == 0:
        features['BOP'] = True
    if settings['eop'] and i >= len(notes) - 1:
        features['EOP'] = True

    # if settings['distance'] != 'none':
    #     for index_offset in range(1, settings['distance_window'] + 1):
    #         left_i = i - index_offset
    #         right_i = i + index_offset
    #         if settings['distance'] == 'integral':
    #             left_tag = "distance:{}".format(left_i)
    #             right_tag = "distance:+{}".format(right_i)
    #             features[left_tag] = integral_distance(notes=notes, from_i=left_i, to_i=i)
    #             features[right_tag] = integral_distance(notes=notes, from_i=i, to_i=right_i)
    #         elif settings['distance'] == 'lattice':
    #             left_x_tag = "x_distance:{}".format(left_i)
    #             right_x_tag = "x_distance:+{}".format(right_i)
    #             left_y_tag = "y_distance:{}".format(left_i)
    #             right_y_tag = "y_distance:+{}".format(right_i)
    #             features[left_x_tag], features[left_y_tag] = lattice_distance(notes=notes, from_i=left_i, to_i=i)
    #             features[right_x_tag], features[right_y_tag] = lattice_distance(notes=notes, from_i=i, to_i=right_i)

    if settings['distance'] == 'integral':
        features['distance:-4'] = c.integral_distance(notes=notes, from_i=i-4, to_i=i)
        features['distance:-3'] = c.integral_distance(notes=notes, from_i=i-3, to_i=i)
        features['distance:-2'] = c.integral_distance(notes=notes, from_i=i-2, to_i=i)
        features['distance:-1'] = c.integral_distance(notes=notes, from_i=i-1, to_i=i)
        features['distance:+1'] = c.integral_distance(notes=notes, from_i=i, to_i=i+1)
        features['distance:+2'] = c.integral_distance(notes=notes, from_i=i, to_i=i+2)
        features['distance:+3'] = c.integral_distance(notes=notes, from_i=i, to_i=i+3)
        features['distance:+4'] = c.integral_distance(notes=notes, from_i=i, to_i=i+4)
    elif settings['distance'] == 'lattice':
        features['x_distance:-4'], features['y_distance:-4'] = c.lattice_distance(notes=notes, from_i=i-4, to_i=i)
        features['x_distance:-3'], features['y_distance:-3'] = c.lattice_distance(notes=notes, from_i=i-3, to_i=i)
        features['x_distance:-2'], features['y_distance:-2'] = c.lattice_distance(notes=notes, from_i=i-2, to_i=i)
        features['x_distance:-1'], features['y_distance:-1'] = c.lattice_distance(notes=notes, from_i=i-1, to_i=i)
        features['x_distance:+1'], features['y_distance:+1'] = c.lattice_distance(notes=notes, from_i=i, to_i=i+1)
        features['x_distance:+2'], features['y_distance:+2'] = c.lattice_distance(notes=notes, from_i=i, to_i=i+2)
        features['x_distance:+3'], features['y_distance:+3'] = c.lattice_distance(notes=notes, from_i=i, to_i=i+3)
        features['x_distance:+4'], features['y_distance:+4'] = c.lattice_distance(notes=notes, from_i=i, to_i=i+4)

    if settings['simple_chording']:
        # Chord features. Approximate with 30 ms offset deltas a la Nakamura.
        left_chord_notes, right_chord_notes = c.chordings(notes=notes, middle_i=i)
        features['left_chord'] = left_chord_notes
        features['right_chord'] = right_chord_notes

    judge_as_chord_trigram = False
    if settings['judge_chords'] and (features['left_chord'] or features['right_chord']):
        judge_as_chord_trigram = True

    if settings['staff']:
        features['staff'] = 0
        if staff == "upper":
            features['staff'] = 1
            # @100: [0.54495717 0.81059147 0.81998371 0.68739401 0.73993751]
            # @1:   [0.54408935 0.80563961 0.82079826 0.6941775  0.73534277]

    if settings['black']:
        features['black_key']: c.black_key(notes, i)

    # if settings['complex_chording']:
        # features['complex_chord'] = complex_chording(notes=notes, annotations=annotations, middle_i=i)

    if settings['leap']:
        # Impact of large leaps? Costs max out, no? Maybe not.
        features['leap'] = 0
        if c.leap_is_excessive(notes, i):
            features['leap'] = 1

    if settings['velocity']:
        oon = notes[i]
        m21_note: note.Note = oon['note']
        on_velocity = m21_note.volume.velocity
        if on_velocity is None:
            on_velocity = 64
        features['velocity'] = on_velocity

    if settings['tempo']:
        tempi = c.tempo_features(notes=notes, middle_i=i)
        for k in tempi:
            features[k] = tempi[k]

    if settings['articulation']:
        arts = c.articulation_features(notes=notes, middle_i=i)
        for k in arts:
            features[k] = arts[k]

    if settings['repeat']:
        reps_before, reps_after = c.repeat_features(notes=notes, middle_i=i)
        features['repeats_before'] = reps_before
        features['repeats_after'] = reps_after

    if settings['judge'] != 'none':
        bad_fingers = c.judgments(judge=judge, notes=notes, middle_i=i, staff=staff)
        for position in bad_fingers:
            for digit in bad_fingers[position]:
                k = "judge_{}:{}".format(digit, position)
                features[k] = bad_fingers[position][digit]
    # FIXME: Lattice distance in Parncutt rules? Approximated by Jacobs.
    #        Mitigated by Balliauw (which just makes the x-distance more
    #        accurate between same-colored keys).

    return features


def phrase2features(notes, staff):
    feature_list = []
    for i in range(len(notes)):
        features = note2features(notes, i, staff)
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


class DExperiment:
    pig_std_dir = {
        'pig_indy': PIG_STD_DIR,
        'pig_seg': PIG_SEGREGATED_STD_DIR,
        'scales': SCALES_STD_PIG_DIR,
        'arpeggios': ARPEGGIOS_STD_PIG_DIR,
        'broken': BROKEN_STD_PIG_DIR,
        'complete_layer_one': COMPLETE_LAYER_ONE_STD_PIG_DIR
    }

    def __init__(self, corpus_names, x=None, y=None, model_version=VERSION,
                 x_train=None, y_train=None, x_test=None, y_test=None):
        self.model_version = model_version
        self.corpus_names = corpus_names
        self.x = x or []
        self.y = y or []
        self.x_train = x_train or []
        self.y_train = y_train or []
        self.x_test = x_test or []
        self.y_test = y_test or []

        self.bad_annot_count = 0
        self.wildcarded_count = 0
        self.good_annot_count = 0
        self.annotated_note_count = 0
        self.total_nondefault_hand_finger_count = 0
        self.total_nondefault_hand_segment_count = 0
        self.test_indices = {}
        self.ordered_test_d_score_titles = []
        self.test_d_scores = {}  # Indexed by score title.

    def test_paths_by_piece(self, corpus=None):
        test_pig_paths = dict()
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            if corpus is not None and corpus_name != corpus:
                continue
            piece_id, annot_id = str(score_title).split('-')
            if piece_id not in test_pig_paths:
                test_pig_paths[piece_id] = list()
            test_pig_path = DExperiment.pig_std_dir[corpus_name] + score_title + PIG_FILE_SUFFIX
            test_pig_paths[piece_id].append(test_pig_path)
        return test_pig_paths

    def test_d_scores_by_piece(self, corpus=None):
        test_d_scores = dict()
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            if corpus is not None and corpus_name != corpus:
                continue
            piece_id, annot_id = str(score_title).split('-')
            if piece_id not in test_d_scores:
                test_d_scores[piece_id] = list()
            test_d_score = self.test_d_scores[score_title]
            test_d_scores[piece_id].append(test_d_score)
        return test_d_scores

    def test_indices_by_piece(self, corpus, piece_id=None):
        index = 0
        done = False
        indices_by_piece = dict()
        if piece_id is not None:
            while not done:
                key = (corpus, piece_id, index)
                if key in sorted(self.test_indices):
                    if piece_id not in indices_by_piece:
                        indices_by_piece[piece_id] = []
                    indices_by_piece[piece_id].append(self.test_indices[key])
                else:
                    done = True
        else:
            for key in sorted(self.test_indices):
                (corpus_name, score_title, annot_index) = key
                the_piece_id, annot_id = str(score_title).split('-')
                if corpus_name == corpus:
                    if the_piece_id not in indices_by_piece:
                        indices_by_piece[the_piece_id] = []
                    indices_by_piece[the_piece_id].append(self.test_indices[key])
        return indices_by_piece

    def predict_and_persist_file(self, test_key, prediction_dir=PREDICTION_DIR):
        (corpus_name, score_title, annot_index) = test_key
        # base_title, annot_id = str(score_title).split('-')
        upper_index, lower_index = self.test_indices[test_key]
        y_pred = my_crf.predict(self.x_test)
        pred_abcdf = "".join(y_pred[upper_index]) + '@' + "".join(y_pred[lower_index])
        pred_annot = DAnnotation(abcdf=pred_abcdf)
        pred_abcdh = ABCDHeader(annotations=[pred_annot])
        pred_d_score = self.test_d_scores[score_title]
        pred_d_score.abcd_header(abcd_header=pred_abcdh)
        pred_pout = PigOut(d_score=pred_d_score)
        pred_pig_path = prediction_dir + score_title + PIG_FILE_SUFFIX
        pred_pout.transform(annotation_index=0, to_file=pred_pig_path)
        return pred_pig_path

    def predict_and_persist(self, prediction_dir=PREDICTION_DIR):
        PigIn.mkdir_if_missing(path_str=prediction_dir, make_missing=True)
        last_base_title = ''
        test_pig_paths = list()
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            base_title, annot_id = str(score_title).split('-')
            if base_title != last_base_title:
                self.predict_and_persist_file(test_key=test_key, prediction_dir=prediction_dir)
                last_base_title = base_title
            test_pig_path = DExperiment.pig_std_dir[corpus_name] + score_title + PIG_FILE_SUFFIX
            test_pig_paths.append(test_pig_path)
        return test_pig_paths

    def get_simple_match_rate(self, prediction_dir=PREDICTION_DIR, output=False):
        """
        Use the executable from Nakamura to calculate SimpleMatchRate.
        """
        pp_path = Path(prediction_dir)
        if not pp_path.is_dir():
            os.makedirs(prediction_dir)
        total_simple_match_count = 0
        total_annot_count = 0
        test_file_count = 0
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            pred_pig_path = self.predict_and_persist_file(test_key=test_key, prediction_dir=prediction_dir)
            test_pig_path = DExperiment.pig_std_dir[corpus_name] + score_title + PIG_FILE_SUFFIX
            match_rate = PigOut.simple_match_rate(gt_pig_path=test_pig_path, pred_pig_path=pred_pig_path)
            total_simple_match_count += match_rate['match_count']
            total_annot_count += match_rate['note_count']
            test_file_count += 1
            # print(result.stdout)
        simple_match_rate = total_simple_match_count / total_annot_count
        if output:
            print("SimpleMatchRate: {}/{} = {}".format(total_simple_match_count, total_annot_count, simple_match_rate))
            print("over {} test files.".format(test_file_count))
        return total_simple_match_count, total_annot_count, simple_match_rate

    def get_my_avg_m_gen(self, prediction_dir=PREDICTION_DIR, test_dir=TEST_DIR, reuse=False, weight=False):
        if reuse:
            PigIn.mkdir_if_missing(path_str=test_dir, make_missing=False)
        else:
            PigIn.mkdir_if_missing(path_str=test_dir, make_missing=True)
            test_pig_paths = self.predict_and_persist(prediction_dir=prediction_dir)
            print("There are {} PIG test paths.".format(len(test_pig_paths)))
            for tpp in test_pig_paths:
                shutil.copy2(tpp, test_dir)
        avg_m_gen, piece_m_gens = PigOut.my_average_m_gen(fingering_files_dir=test_dir,
                                                          prediction_input_dir=prediction_dir, weight=weight)
        return avg_m_gen, piece_m_gens

    def get_my_avg_m(self, prediction_dir=PREDICTION_DIR, test_dir=TEST_DIR, reuse=False, weight=False):
        if reuse:
            PigIn.mkdir_if_missing(path_str=test_dir, make_missing=False)
        else:
            PigIn.mkdir_if_missing(path_str=test_dir, make_missing=True)
            test_pig_paths = self.predict_and_persist(prediction_dir=prediction_dir)
            print("There are {} PIG test paths.".format(len(test_pig_paths)))
            for tpp in test_pig_paths:
                shutil.copy2(tpp, test_dir)
        avg_m, piece_ms = PigOut.my_average_m(fingering_files_dir=test_dir,
                                              prediction_input_dir=prediction_dir, weight=weight)
        return avg_m, piece_ms

    def get_complex_match_rates(self, weight=False, prediction_dir=PREDICTION_DIR, output=False):
        PigIn.mkdir_if_missing(path_str=PREDICTION_DIR, make_missing=True)
        y_pred = my_crf.predict(self.x_test)
        total_note_count = 0
        d_score_count = 0
        pred_pig_path = ''
        combined_match_rates = {}
        piece_data = dict()
        for corpus_name in CORPUS_NAMES:
            test_indices = self.test_indices_by_piece(corpus=corpus_name)
            test_pig_paths = self.test_paths_by_piece(corpus=corpus_name)
            test_d_scores = self.test_d_scores_by_piece(corpus=corpus_name)
            tested_pieces = dict()
            for piece_id in test_pig_paths:
                note_count = 0
                annot_count = 0
                if piece_id not in tested_pieces:
                    upper_index, lower_index = test_indices[piece_id][0]
                    pred_abcdf = "".join(y_pred[upper_index]) + '@' + "".join(y_pred[lower_index])
                    pred_annot = DAnnotation(abcdf=pred_abcdf)
                    pred_abcdh = ABCDHeader(annotations=[pred_annot])
                    pred_d_score = test_d_scores[piece_id][0]
                    d_score_count += 1
                    pred_d_score.abcd_header(abcd_header=pred_abcdh)
                    pred_pout = PigOut(d_score=pred_d_score)
                    file_id = "{}-1".format(piece_id)
                    pred_pig_path = prediction_dir + file_id + PIG_FILE_SUFFIX
                    pred_pig_content = pred_pout.transform(annotation_index=0, to_file=pred_pig_path)
                    note_count = pred_d_score.note_count()
                    annot_count = note_count * len(test_pig_paths[piece_id])
                match_rates = PigOut.single_prediction_complex_match_rates(gt_pig_paths=test_pig_paths[piece_id],
                                                                           pred_pig_path=pred_pig_path)
                total_note_count += note_count
                piece_data[piece_id] = dict()
                for key in match_rates:
                    match_count = round(match_rates[key] * annot_count)
                    raw_rate = match_rates[key]
                    piece_data[piece_id][key] = {
                        'match_count': match_count,
                        'note_count': note_count,
                        'annot_count': annot_count,
                        'raw_rate': raw_rate
                    }
                    if weight:
                        match_rates[key] *= note_count
                    if key not in combined_match_rates:
                        combined_match_rates[key] = 0
                    combined_match_rates[key] += match_rates[key]

        for key in match_rates:
            if weight:
                combined_match_rates[key] /= total_note_count
            else:
                combined_match_rates[key] /= d_score_count
        return combined_match_rates, piece_data


#####################################################
# MAIN BLOCK
#####################################################
creal = Corporeal()
judge_model_name = c.VERSION_FEATURES[VERSION]['judge']
judge = None
if judge_model_name != 'none':
    judge = creal.get_model(judge_model_name)

corpora_str = "-".join(CORPUS_NAMES)
experiment_name = corpora_str + '__' + TEST_METHOD + '__' + VERSION
ex = c.unpickle_it(obj_type="DExperiment", file_name=experiment_name)
if ex is None:
    ex = DExperiment(corpus_names=CORPUS_NAMES)
    for corpus_name in CORPUS_NAMES:
        da_corpus = c.unpickle_it(obj_type="DCorpus", file_name=corpus_name, use_dill=True)
        if da_corpus is None:
            da_corpus = creal.get_corpus(corpus_name=corpus_name)
            c.pickle_it(obj=da_corpus, obj_type="DCorpus", file_name=corpus_name, use_dill=True)
        for da_score in da_corpus.d_score_list():
            abcdh = da_score.abcd_header()
            annot_count = abcdh.annotation_count()
            annot = da_score.annotation_by_index(index=0)
            segger = ManualDSegmenter(level='.', d_annotation=annot)
            da_score.segmenter(segger)
            da_unannotated_score = copy.deepcopy(da_score)
            score_title = da_score.title()
            # if score_title != 'Sonatina 4.1':
            # continue
            for staff in STAFFS:
                print("Processing {} staff of {} score from {} corpus.".format(staff, score_title, corpus_name))
                ordered_offset_note_segments = da_score.ordered_offset_note_segments(staff=staff)
                seg_count = len(ordered_offset_note_segments)
                for annot_index in range(annot_count):
                    annot = da_score.annotation_by_index(annot_index)
                    authority = annot.authority()
                    hsd_segments = segger.segment_annotation(annotation=annot, staff=staff)
                    seg_index = 0
                    for hsd_seg in hsd_segments:
                        ordered_notes = ordered_offset_note_segments[seg_index]
                        note_len = len(ordered_notes)
                        seg_len = len(hsd_seg)
                        seg_index += 1
                        if note_len != seg_len:
                            print("Bad annotation by {} for score {}. Notes: {} Fingers: {}".format(
                                authority, score_title, note_len, seg_len))
                            ex.bad_annot_count += 1
                            continue
                        nondefault_hand_finger_count = nondefault_hand_count(hsd_seq=hsd_seg, staff=staff)
                        if nondefault_hand_finger_count:
                            ex.total_nondefault_hand_segment_count += 1
                            print("Non-default hand specified by annotator {} in score {}: {}".format(
                                authority, score_title, hsd_seg))
                            ex.total_nondefault_hand_finger_count += nondefault_hand_finger_count
                            if SEGREGATE_HANDS:
                                ex.bad_annot_count += 1
                                continue
                        if has_wildcard(hsd_seq=hsd_seg):
                            # print("Wildcard disallowed from annotator {} in score {}: {}".format(
                                # authority, score_title, hsd_seg))
                            ex.wildcarded_count += 1
                            continue
                        ex.annotated_note_count += note_len
                        ex.x.append(phrase2features(ordered_notes, staff))
                        ex.y.append(phrase2labels(hsd_seg))
                        if c.has_preset_evaluation_defined(corpus_name=corpus_name):
                            if c.is_in_test_set(title=score_title, corpus_name=corpus_name):
                                test_key = (corpus_name, score_title, annot_index)
                                if test_key not in ex.test_indices:
                                    ex.test_indices[test_key] = []
                                ex.test_indices[test_key].append(len(ex.y_test))
                                ex.x_test.append(phrase2features(ordered_notes, staff))
                                ex.y_test.append(phrase2labels(hsd_seg))
                                if staff == "upper" and annot_index == 0:
                                    ex.ordered_test_d_score_titles.append(da_score)
                                    ex.test_d_scores[score_title] = da_unannotated_score
                            else:
                                ex.x_train.append(phrase2features(ordered_notes, staff))
                                ex.y_train.append(phrase2labels(hsd_seg))
                        ex.good_annot_count += 1
    c.pickle_it(obj=ex, obj_type="DExperiment", file_name=experiment_name)

# print(token_lists)
print("Example count: {}".format(len(ex.x)))
if TEST_METHOD == 'preset':
    print("Training count: {}".format(len(ex.y_train)))
    print("Test count: {}".format(len(ex.y_test)))
print("Good examples: {}".format(ex.good_annot_count))
print("Bad examples: {}".format(ex.bad_annot_count))
print("Wildcarded examples: {}".format(ex.wildcarded_count))
print("Total annotated notes: {}".format(ex.annotated_note_count))
print("Total nondefault hand fingerings: {}".format(ex.total_nondefault_hand_finger_count))
print("Total nondefault hand phrases: {}".format(ex.total_nondefault_hand_segment_count))

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
    VERSION, TEST_METHOD, corpora_str))
print("Clean list: {}".format(list(c.CLEAN_LIST.keys())))
