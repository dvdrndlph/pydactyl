__author__ = 'David Randolph'
# Copyright (c) 2021, 2022 David A. Randolph.
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
import copy
import pprint
from datetime import datetime
import shutil
import os
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from pathlib import Path
import pydactyl.util.CrfUtil as c
from pydactyl.util.DExperimentOpts import DExperimentOpts
from pydactyl.eval.Corporeal import Corporeal
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
from pydactyl.dcorpus.DAnnotation import DAnnotation
from pydactyl.dcorpus.ABCDHeader import ABCDHeader
from pydactyl.dcorpus.PigInOut import PigIn, PigOut, PIG_STD_DIR, PIG_FILE_SUFFIX, PIG_SEGREGATED_STD_DIR
from pydactyl.eval.Corporeal import ARPEGGIOS_STD_PIG_DIR, SCALES_STD_PIG_DIR, BROKEN_STD_PIG_DIR, COMPLETE_LAYER_ONE_STD_PIG_DIR


class DExperiment:
    pig_std_dir = {
        'pig_indy': PIG_STD_DIR,
        'pig_seg': PIG_SEGREGATED_STD_DIR,
        'scales': SCALES_STD_PIG_DIR,
        'arpeggios': ARPEGGIOS_STD_PIG_DIR,
        'broken': BROKEN_STD_PIG_DIR,
        'complete_layer_one': COMPLETE_LAYER_ONE_STD_PIG_DIR
    }

    def __init__(self, opts: DExperimentOpts,
                 x: list = None, y: list = None,
                 x_train: list = None, y_train: list = None,
                 x_test: list = None, y_test: list = None,
                 as_features=True):
        self.opts = opts
        self.model_features = opts.model_features
        self.model_version = opts.model_version
        self.test_method = opts.test_method
        self.fold_count = opts.fold_count
        self.corpus_names = opts.corpus_names
        self.staffs = opts.staffs
        self.segregate_hands = opts.segregate_hands
        self.default_prediction_dir = '/tmp/crf' + self.model_version + 'prediction/'
        self.default_test_dir = '/tmp/crf' + self.model_version + 'test/'
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
        self.test_indices = {}  # Indexed by test_key tuple (corpus_name, score_title, annotation_index)
        self.y_test_keys = {}  # Hash of y_test integer indices to test keys.

        self.test_d_scores = {}  # Indexed by score title. Implies score titles unique across corpora. Seems bad.
        # This all seems like a big kludge.

        self.as_features = as_features
        self.note_func = opts.note_func
        self.reverse = opts.reverse

    def corpora_name(self):
        corpora_str = "-".join(self.corpus_names)
        return corpora_str

    def experiment_name(self):
        corpora_str = self.corpora_name()
        file_name = corpora_str + '__' + self.test_method + '__' + self.model_version
        return file_name

    def evaluate_trained_model(self, the_model, x_test=None, y_test=None):
        if y_test is None:
            x_test = self.x_test
            y_test = self.y_test
        labels = list(the_model.classes_)
        print(labels)
        y_predicted = the_model.predict(x_test)
        flat_weighted_f1 = metrics.flat_f1_score(y_test, y_predicted, average='weighted', labels=labels)
        print("Flat weighted F1: {}".format(flat_weighted_f1))

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(y_test, y_predicted, labels=sorted_labels, digits=4))
        return y_predicted

    def train_and_evaluate(self, the_model, x_train=None, y_train=None, x_test=None, y_test=None):
        if y_train is None:
            x_train = self.x_train
            y_train = self.y_train
        the_model.fit(x_train, y_train)
        return self.evaluate_trained_model(the_model=the_model, x_test=x_test, y_test=y_test)

    def split_and_evaluate(self, the_model, test_size=0.4, random_state=0):
        split_x_train, split_x_test, split_y_train, split_y_test = \
            train_test_split(self.x, self.y, test_size=0.4, random_state=0)
        self.train_and_evaluate(the_model=the_model, x_train=split_x_train, y_train=split_y_train,
                                x_test=split_x_test, y_test=split_y_test)

    def evaluate(self, the_model, is_trained):
        start_dt = datetime.now()
        if self.test_method == 'cross-validate':
            scores = cross_val_score(the_model, self.x, self.y, cv=self.fold_count)
            # scores = cross_validate(my_crf, ex.x, ex.y, cv=5, scoring="flat_precision_score")
            print(scores)
            avg_score = sum(scores) / len(scores)
            print("Average cross-validation score: {}".format(avg_score))
        elif self.test_method == 'preset':
            if is_trained:
                predictions = self.evaluate_trained_model(the_model=the_model)
            else:
                predictions = self.train_and_evaluate(the_model=the_model)
            self.direct_avg_m(predictions=predictions)
            # total_simple_match_count, total_annot_count, simple_match_rate = \
            #     self.get_simple_match_rate(predictions=predictions, output=True)
            # print("Simple match rate: {}".format(simple_match_rate))
            # result, complex_piece_results = ex.get_complex_match_rates(predictions=predictions, weight=False)
            # print("Unweighted avg M for crf{} over {}: {}".format(model.CRF_VERSION, CORPUS_NAMES, result))

            # result, my_piece_results = self.get_my_avg_m(predictions=predictions, weight=False, reuse=False)
            # print("My unweighted avg M for crf{} over {}: {}".format(self.model_version, self.corpus_names, result))

            # for key in sorted(complex_piece_results):
            # print("nak {} => {}".format (key, complex_piece_results[key]))
            # print(" my {} => {}".format(key, my_piece_results[key]))
            # print("")
            # result, piece_results = ex.get_complex_match_rates(weight=True)
            # print("Weighted avg M for crf{} over {}: {}".format(model.CRF_VERSION, CORPUS_NAMES, result))

            # result, piece_results = self.get_my_avg_m(predictions=predictions, weight=True, reuse=True)
            # print("Weighted avg m_gen for crf{} over {}: {}".format(self.model_version, self.corpus_names, result))
        else:
            self.split_and_evaluate(the_model=the_model, test_size=0.4, random_state=0)
        print("Run of crf model {} against {} test set over {} corpus has completed successfully.".format(
            self.model_version, self.test_method, self.corpora_name()))
        end_dt = datetime.now()
        execution_duration_minutes = (end_dt - start_dt)
        trained_prefix = 'un'
        if is_trained:
            trained_prefix = ''
        print("Total running time (wall clock) for {}trained model: {}".format(
            trained_prefix, execution_duration_minutes))

    def phrase2features(self, notes: list, staff):
        feature_list = []
        for i in range(len(notes)):
            features = self.note_func(notes, i, staff)
            feature_list.append(features)
        if self.reverse:
            feature_list.reverse()
        return feature_list

    def phrase2attrs(self, notes, staff):
        return self.phrase2features(notes, staff)

    def phrase2labels(self, handed_strike_digits: list):
        if self.reverse:
            handed_strike_digits.reverse()
        return handed_strike_digits

    def append_example(self, ordered_notes, staff, hsd_seg, is_train=False,
                       is_test=False, test_key=None, d_score=None):
        # IMPORTANT: An example here is segregated, representing one hand (or one staff).
        note_len = len(ordered_notes)
        self.annotated_note_count += note_len
        self.x.append(self.phrase2features(ordered_notes, staff))
        self.y.append(self.phrase2labels(hsd_seg))
        if is_test:
            score_title = d_score.title()
            if test_key not in self.test_indices:
                self.test_indices[test_key] = []
            self.test_indices[test_key].append(self.good_annot_count)
            self.y_test_keys[self.good_annot_count] = test_key
            self.x_test.append(self.phrase2features(ordered_notes, staff))
            self.y_test.append(self.phrase2labels(hsd_seg))
            self.test_d_scores[score_title] = d_score
        elif is_train:
            self.x_train.append(self.phrase2features(ordered_notes, staff))
            self.y_train.append(self.phrase2labels(hsd_seg))
        self.good_annot_count += 1

    def print_summary(self):
        print("Example count: {}".format(len(self.x)))
        if self.test_method == 'preset':
            print("Training count: {}".format(len(self.y_train)))
            print("Test count: {}".format(len(self.y_test)))
        print("Good examples: {}".format(self.good_annot_count))
        print("Bad examples: {}".format(self.bad_annot_count))
        print("Wildcarded examples: {}".format(self.wildcarded_count))
        print("Total annotated notes: {}".format(self.annotated_note_count))
        print("Total nondefault hand fingerings: {}".format(self.total_nondefault_hand_finger_count))
        print("Total nondefault hand phrases: {}".format(self.total_nondefault_hand_segment_count))

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

    def persist_prediction_to_pig_file(self, test_key, predictions, prediction_dir=None):
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        (corpus_name, score_title, annot_index) = test_key
        # base_title, annot_id = str(score_title).split('-')
        upper_index, lower_index = self.test_indices[test_key]
        pred_abcdf = "".join(predictions[upper_index]) + '@' + "".join(predictions[lower_index])
        pred_annot = DAnnotation(abcdf=pred_abcdf)
        pred_abcdh = ABCDHeader(annotations=[pred_annot])
        pred_d_score = self.test_d_scores[score_title]
        pred_d_score.abcd_header(abcd_header=pred_abcdh)
        pred_pout = PigOut(d_score=pred_d_score)
        pred_pig_path = prediction_dir + score_title + PIG_FILE_SUFFIX
        pred_pout.transform(annotation_index=0, to_file=pred_pig_path)
        return pred_pig_path

    def predict_and_persist(self, predictions, prediction_dir=None):
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        PigIn.mkdir_if_missing(path_str=prediction_dir, make_missing=True)
        last_base_title = ''
        test_pig_paths = list()
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            base_title, annot_id = str(score_title).split('-')
            if base_title != last_base_title:
                # We only get to predict once per piece in Nakamura evaluation.
                self.persist_prediction_to_pig_file(test_key=test_key,
                                                    predictions=predictions, prediction_dir=prediction_dir)
                last_base_title = base_title
            test_pig_path = DExperiment.pig_std_dir[corpus_name] + score_title + PIG_FILE_SUFFIX
            test_pig_paths.append(test_pig_path)
        return test_pig_paths

    def persist_predictions(self, predictions, prediction_dir=None):
        # predictions is a list of lists of strike fingers, aligned with
        # the y_test data.
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        PigIn.mkdir_if_missing(path_str=prediction_dir, make_missing=True)
        test_pig_paths = list()
        pred_pig_paths = list()
        guesses = list()
        last_test_key = ''
        test_index = 0
        for guess in predictions:
            # We have guesses about 1 or 2 staves. They are interleaved in the predictions list.
            # If we have two, both need to make it to the PIG file.
            guesses.append(guess)
            test_key = self.y_test_keys[test_index]
            (corpus_name, score_title, annot_index) = test_key
            base_title, annot_id = str(score_title).split('-')
            if last_test_key and test_key != last_test_key:
                prediction_pig_path = self.persist_prediction_to_pig_file(test_key=test_key, predictions=guesses,
                                                                          prediction_dir=prediction_dir)
                pred_pig_paths.append(prediction_pig_path)
                test_pig_path = DExperiment.pig_std_dir[corpus_name] + score_title + PIG_FILE_SUFFIX
                test_pig_paths.append(test_pig_path)
                guesses = []
            last_base_title = base_title
        return test_pig_paths, pred_pig_paths

    def get_simple_match_rate_for_predictions(self, predictions, prediction_dir=None, output=False):
        """
        Use the executable from Nakamura to calculate SimpleMatchRate.
        """
        test_paths, pred_paths = self.persist_predictions(predictions=predictions, prediction_dir=prediction_dir)
        test_path_count = len(test_paths)
        pred_path_count = len(test_paths)
        if test_path_count != pred_path_count:
            raise Exception("Mismatched PIG file counts.")

        total_simple_match_count = 0
        total_annot_count = 0
        test_file_count = 0
        for i in range(test_path_count):
            pred_pig_path = pred_paths[i]
            test_pig_path = test_paths[i]
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

    def get_simple_match_rate(self, predictions, prediction_dir=None, output=False):
        """
        Use the executable from Nakamura to calculate SimpleMatchRate.
        """
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        pp_path = Path(prediction_dir)
        if not pp_path.is_dir():
            os.makedirs(prediction_dir)
        total_simple_match_count = 0
        total_annot_count = 0
        test_file_count = 0
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            pred_pig_path = self.persist_prediction_to_pig_file(predictions=predictions, test_key=test_key,
                                                                prediction_dir=prediction_dir)
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

    def get_my_avg_m_gen(self, prediction_dir=None, test_dir=None, reuse=False, weight=False):
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        if test_dir is None:
            test_dir = self.default_test_dir
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

    @staticmethod
    def match_count(predicted, ground_truth):
        pred_len = len(predicted)
        gt_len = len(ground_truth)
        if gt_len == 0:
            raise Exception("Phrase is zero length.")
        if gt_len != pred_len:
            raise Exception("Counts do not match.")
        match_count = 0
        for i in range(pred_len):
            if predicted[i] == ground_truth[i]:
                match_count += 1
        return match_count

    @staticmethod
    def soft_match_count(predicted, ground_truths):
        pred_len = len(predicted)
        gt_len = len(ground_truths[0])
        if gt_len == 0:
            raise Exception("Phrase is zero length.")
        if gt_len != pred_len:
            raise Exception("Counts do not match.")
        match_bits = [0 for i in range(pred_len)]
        for ground_truth in ground_truths:
            for i in range(pred_len):
                if predicted[i] == ground_truth[i]:
                    match_bits[i] = 1
        match_count = sum(match_bits)
        return match_count

    @staticmethod
    def high_match_count(predicted, ground_truths):
        pred_len = len(predicted)
        gt_len = len(ground_truths[0])
        if gt_len == 0:
            raise Exception("Phrase is zero length.")
        if gt_len != pred_len:
            raise Exception("Counts do not match.")
        max_count = 0
        for ground_truth in ground_truths:
            match_bits = [0 for i in range(pred_len)]
            for i in range(pred_len):
                if predicted[i] == ground_truth[i]:
                    match_bits[i] = 1
            match_count = sum(match_bits)
            if match_count > max_count:
                max_count = match_count
        return max_count

    def direct_avg_m(self, predictions, weight=False):
        # predictions contain predictions for all items in the test set.
        # We need to pull the first "exemplar" for each score and compare
        # it to every ground truth we have for the score.
        last_base_title = ''
        test_keys_for_base_title = dict()
        prediction_key_for_base_title = dict()
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            base_title, annot_id = str(score_title).split('-')
            if base_title != last_base_title:
                prediction_key_for_base_title[base_title] = test_key
                test_keys_for_base_title[base_title] = list()
                # We only get to predict once per piece in Nakamura evaluation.
                last_base_title = base_title
            test_keys_for_base_title[base_title].append(test_key)
        total_upper_note_count = 0
        total_lower_note_count = 0
        total_upper_match_count = 0
        total_lower_match_count = 0
        score_note_counts = dict()
        score_gen_matches = dict()
        base_title_match_rates = dict()
        base_title_note_counts = dict()
        for base_title in prediction_key_for_base_title:
            total_score_upper_note_count = 0
            total_score_lower_note_count = 0
            total_score_upper_match_count = 0
            total_score_lower_match_count = 0
            pred_key = prediction_key_for_base_title[base_title]
            (pred_upper_i, pred_lower_i) = self.test_indices[pred_key]
            pred_upper = predictions[pred_upper_i]
            pred_lower = predictions[pred_lower_i]
            upper_note_count = len(pred_upper)
            lower_note_count = len(pred_lower)
            combined_note_count = upper_note_count + lower_note_count
            if base_title not in base_title_match_rates:
                base_title_match_rates[base_title] = {
                    'upper': {
                        'gen': 0.0,
                        'high': 0.0,
                        'soft': 0.0
                    },
                    'lower': {
                        'gen': 0.0,
                        'high': 0.0,
                        'soft': 0.0
                    },
                    'combined': {
                        'gen': 0.0,
                        'high': 0.0,
                        'soft': 0.0
                    },
                }
                base_title_note_counts[base_title] = {
                    'upper': upper_note_count,
                    'lower': lower_note_count,
                    'combined': combined_note_count
                }
            score_pred_fingering = {
                'upper': pred_upper,
                'lower': pred_lower,
                'combined': pred_upper + pred_lower
            }
            score_test_fingrings = {
                'upper': list(),
                'lower': list(),
                'combined': list()
            }
            for test_key in test_keys_for_base_title[base_title]:
                # print("Compare {} to {}".format(pred_key, test_key))
                (test_upper_i, test_lower_i) = self.test_indices[test_key]
                test_upper = self.y_test[test_upper_i]
                score_test_fingrings['upper'].append(test_upper)
                upper_match_count = DExperiment.match_count(predicted=pred_upper, ground_truth=test_upper)
                total_upper_match_count += upper_match_count
                total_upper_note_count += upper_note_count
                total_score_upper_match_count += upper_match_count
                total_score_upper_note_count += upper_note_count

                test_lower = self.y_test[test_lower_i]
                score_test_fingrings['lower'].append(test_lower)
                score_test_fingrings['combined'].append(test_upper + test_lower)
                lower_match_count = DExperiment.match_count(predicted=pred_lower, ground_truth=test_lower)
                total_lower_match_count += lower_match_count
                total_lower_note_count += lower_note_count
                total_score_lower_note_count += lower_note_count
                total_score_lower_match_count += lower_match_count
            total_score_match_count = total_score_upper_match_count + total_score_lower_match_count
            total_score_note_count = total_score_upper_note_count + total_score_lower_note_count
            base_title_match_rates[base_title]['combined']['gen'] = total_score_match_count / total_score_note_count
            base_title_match_rates[base_title]['upper']['gen'] = total_score_upper_match_count / total_score_upper_note_count
            base_title_match_rates[base_title]['lower']['gen'] = total_score_lower_match_count / total_score_lower_note_count

            score_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['combined'],
                                                                  score_test_fingrings['combined'])
            base_title_match_rates[base_title]['combined']['soft'] = score_soft_match_count / combined_note_count
            score_upper_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['upper'],
                                                                        score_test_fingrings['upper'])
            base_title_match_rates[base_title]['upper']['soft'] = score_upper_soft_match_count / upper_note_count
            score_lower_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['lower'],
                                                                        score_test_fingrings['lower'])
            base_title_match_rates[base_title]['lower']['soft'] = score_lower_soft_match_count / lower_note_count

            score_high_match_count = DExperiment.high_match_count(score_pred_fingering['combined'],
                                                                  score_test_fingrings['combined'])
            base_title_match_rates[base_title]['combined']['high'] = score_high_match_count / combined_note_count
            score_upper_high_match_count = DExperiment.high_match_count(score_pred_fingering['upper'],
                                                                        score_test_fingrings['upper'])
            base_title_match_rates[base_title]['upper']['high'] = score_upper_high_match_count / upper_note_count
            score_lower_high_match_count = DExperiment.high_match_count(score_pred_fingering['lower'],
                                                                        score_test_fingrings['lower'])
            base_title_match_rates[base_title]['lower']['high'] = score_lower_high_match_count / lower_note_count

        pprint.pprint(base_title_match_rates)

        sums_of_rates = {
            'upper': {
                'gen': 0.0,
                'high': 0.0,
                'soft': 0.0
            },
            'lower': {
                'gen': 0.0,
                'high': 0.0,
                'soft': 0.0
            },
            'combined': {
                'gen': 0.0,
                'high': 0.0,
                'soft': 0.0
            },
        }
        weighted_sums_of_rates = copy.deepcopy(sums_of_rates)
        m_rates = copy.deepcopy(sums_of_rates)
        weighted_m_rates = copy.deepcopy(sums_of_rates)
        total_note_counts = {
            'upper': 0,
            'lower': 0,
            'combined': 0
        }
        total_weights = copy.deepcopy(total_note_counts)
        title_count = len(base_title_match_rates)
        for base_title, rates in base_title_match_rates.items():
            for staff in ('upper', 'lower', 'combined'):
                for method in ('gen', 'high', 'soft'):
                    sums_of_rates[staff][method] += rates[staff][method]
                    weighted_sums_of_rates[staff][method] += rates[staff][method] * base_title_note_counts[base_title][staff]
                total_weights[staff] += base_title_note_counts[base_title][staff]

        for staff in ('upper', 'lower', 'combined'):
            for method in ('gen', 'high', 'soft'):
                m_rates[staff][method] = sums_of_rates[staff][method] / title_count
                weighted_m_rates[staff][method] = weighted_sums_of_rates[staff][method] / total_weights[staff]

        total_note_count = total_upper_note_count + total_lower_note_count
        total_match_count = total_upper_match_count + total_lower_match_count
        upper_smr = total_upper_match_count / total_upper_note_count
        lower_smr = total_lower_match_count / total_lower_note_count
        total_smr = total_match_count / total_note_count
        m_rates['upper']['simple'] = upper_smr
        m_rates['lower']['simple'] = lower_smr
        m_rates['combined']['simple'] = total_smr
        # The simple match rate is implicitly weighted, as it is just total matches over total notes.
        weighted_m_rates['upper']['simple'] = upper_smr
        weighted_m_rates['lower']['simple'] = lower_smr
        weighted_m_rates['combined']['simple'] = total_smr

        pprint.pprint(m_rates)
        pprint.pprint(weighted_m_rates)

        return m_rates, weighted_m_rates



    def get_my_avg_m(self, predictions, prediction_dir=None, test_dir=None, reuse=False, weight=False):
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        if test_dir is None:
            test_dir = self.default_test_dir
        if reuse:
            PigIn.mkdir_if_missing(path_str=test_dir, make_missing=False)
        else:
            PigIn.mkdir_if_missing(path_str=test_dir, make_missing=True)
            test_pig_paths = self.predict_and_persist(predictions=predictions, prediction_dir=prediction_dir)
            print("There are {} PIG test paths.".format(len(test_pig_paths)))
            for tpp in test_pig_paths:
                shutil.copy2(tpp, test_dir)
        avg_m, piece_ms = PigOut.my_average_m(fingering_files_dir=test_dir,
                                              prediction_input_dir=prediction_dir, weight=weight)
        return avg_m, piece_ms

    def get_complex_match_rates(self, crf, weight=False, prediction_dir=None, output=False):
        if prediction_dir is None:
            prediction_dir = self.default_prediction_dir
        PigIn.mkdir_if_missing(path_str=prediction_dir, make_missing=True)
        y_pred = crf.predict(self.x_test)
        total_note_count = 0
        d_score_count = 0
        pred_pig_path = ''
        combined_match_rates = {}
        piece_data = dict()
        match_rates = dict()
        for corpus_name in self.corpus_names:
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

    def load_data(self, clean_list):
        creal = Corporeal()
        experiment_name = self.experiment_name()

        for corpus_name in self.corpus_names:
            da_corpus = c.unpickle_it(obj_type="DCorpus", clean_list=clean_list,
                                      file_name=corpus_name, use_dill=True)
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
                for staff in self.staffs:
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
                                self.bad_annot_count += 1
                                continue
                            nondefault_hand_finger_count = c.nondefault_hand_count(hsd_seq=hsd_seg, staff=staff)
                            if nondefault_hand_finger_count:
                                self.total_nondefault_hand_segment_count += 1
                                print("Non-default hand specified by annotator {} in score {}: {}".format(
                                    authority, score_title, hsd_seg))
                                self.total_nondefault_hand_finger_count += nondefault_hand_finger_count
                                if self.segregate_hands:
                                    self.bad_annot_count += 1
                                    continue
                            if c.has_wildcard(hsd_seq=hsd_seg):
                                # print("Wildcard disallowed from annotator {} in score {}: {}".format(
                                # authority, score_title, hsd_seg))
                                self.wildcarded_count += 1
                                continue
                            if c.has_preset_evaluation_defined(corpus_name=corpus_name):
                                if c.is_in_test_set(title=score_title, corpus_name=corpus_name):
                                    test_key = (corpus_name, score_title, annot_index)
                                    self.append_example(ordered_notes, staff, hsd_seg, is_test=True,
                                                        test_key=test_key, d_score=da_unannotated_score)
                                else:
                                    self.append_example(ordered_notes, staff, hsd_seg, is_train=True)
                            else:
                                self.append_example(ordered_notes, staff, hsd_seg)
        print("Data loaded. Clean list: {}".format(list(clean_list.keys())))
        return experiment_name
