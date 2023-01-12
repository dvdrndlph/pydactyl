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

import shutil
import os
from pydactyl.util.CrfUtil import note2attrs, note2features
from pathlib import Path
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

    def __init__(self, corpus_names, model_version, x=None, y=None,
                 x_train=None, y_train=None, x_test=None, y_test=None,
                 as_features=True, note_func=note2features):
        self.model_version = model_version
        self.default_prediction_dir = '/tmp/crf' + model_version + 'prediction/'
        self.default_test_dir = '/tmp/crf' + model_version + 'test/'
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
        self.test_indices = {}  # Indexed by test_key tuple (corpus_name, score_title, annotation_index)
        self.y_test_keys = {}  # Hash of y_test integer indices to test keys.

        self.test_d_scores = {}  # Indexed by score title. Implies score titles unique across corpora. Seems bad.
        # This all seems like a big kludge.

        self.as_features = as_features
        self.note_func = note_func

    def phrase2features(self, notes, staff):
        feature_list = []
        for i in range(len(notes)):
            features = self.note_func(notes, i, staff)
            feature_list.append(features)
        return feature_list

    def phrase2attrs(self, notes, staff):
        return self.phrase2features(notes, staff)

    def phrase2labels(self, handed_strike_digits):
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

    def print_summary(self, test_method):
        print("Example count: {}".format(len(self.x)))
        if test_method == 'preset':
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
                prediction_pig_path = self.persist_prediction_to_pig_file(test_key=test_key, prediction=guesses,
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
