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
import re
from datetime import datetime
from dataclasses import dataclass
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import make_scorer
import pydactyl.crf.CrfUtil as c
from pydactyl.eval.DExperimentOpts import DExperimentOpts
from pydactyl.dcorpus.DNotesData import DNotesData
from pydactyl.eval.Corporeal import Corporeal
from pydactyl.dcorpus.OneDSegmenter import OneDSegmenter
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
from pydactyl.dcorpus.PigInOut import PigIn, PigOut, PIG_STD_DIR, PIG_FILE_SUFFIX, PIG_SEGREGATED_STD_DIR
from pydactyl.eval.Corporeal import ARPEGGIOS_STD_PIG_DIR, SCALES_STD_PIG_DIR, BROKEN_STD_PIG_DIR, COMPLETE_LAYER_ONE_STD_PIG_DIR


@dataclass(frozen=True)
class ScoreKey:
    corpus_name: str
    score_title: str

    def __eq__(self, other):
        if not isinstance(other, ScoreKey):
            raise Exception("ScoreKey can only be compared to another ScoreKey.")
        if self.corpus_name == other.corpus_name and self.score_title == other.score_title:
            return True
        return False


@dataclass(frozen=True)
class SegmentKey:
    staff: str
    score_key: ScoreKey
    seg_index: int = 0

    def __eq__(self, other):
        if not isinstance(other, SegmentKey):
            raise Exception("SegmentKey can only be compared to another SegmentKey.")
        if (self.score_key == other.score_key and
                self.staff == other.staff and self.seg_index == other.seg_index):
            return True
        return False

    def score_key(self):
        return self.score_key


@dataclass(frozen=True)
class ExampleKey:
    seg_key: SegmentKey
    annot_index: int = 0

    def __eq__(self, other):
        if not isinstance(other, ExampleKey):
            raise Exception("ExampleKey can only be compared to another ExampleKey.")
        if self.seg_key == other.seg_key and self.annot_index == other.annot_index:
            return True
        return False

    def score_key(self):
        return self.seg_key.score_key


class AnnotatedExample:
    def __init__(self, seg_key: SegmentKey, notes, hsd_seg, annot_index, d_score):
        self.example_key = ExampleKey(seg_key=seg_key, annot_index=annot_index)
        self.hsd_seg = hsd_seg
        self.notes = notes
        self.seg_key = seg_key
        self.annot_index = annot_index
        self.d_score = d_score

    def score_key(self):
        return self.seg_key.score_key()


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
        self.pickling = opts.pickling
        self.consonance_threshold = opts.consonance_threshold
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
        self.test_note_count = 0
        self.train_note_count = 0

        self.y_staff = {}  # Map example index to associated staff.
        self.y_score_key = {}  # Map example index to tuple of associated corpus name and DScore title.
        self.y_test_y_index = {}  # Map the test example index back to its index in y.

        # Keep track of groups for k-fold cross-validation.
        self.group_id = {}  # Map info tuples to integer group identifiers.
        self.group_note_count = {}  # Map group identifiers to counts of notes contained therein.
        self.group_score_key = {}  # Map group identifiers to counts of notes contained therein.
        self.group_assignments = []  # The group identifier assigned to each example.
        self.train_group_assignments = []

        self.bad_annot_count = 0
        self.wildcarded_count = 0
        self.good_annot_count = 0
        self.annotated_note_count = 0
        self.total_nondefault_hand_finger_count = 0
        self.total_nondefault_hand_segment_count = 0

        # FIXME: marked for death?
        self.test_indices = {}  # Indexed by test_key tuple (corpus_name, score_title, annotation_index)
                                # Each test key tuple will be mapped to an array of 1-2 index values,
                                # one for each staff, though single staff work is probably pretty hosed up by now.
        self.test_key = {}  # Hash of test integer indices to test keys.
        self.example_indices = {}  # Map (corpus_name, score_title, annot_index) tuple to 1-2 indexes in to x and y.
        self.example_key = {}  # Map example integer of example in x and y to its corresponding tuple.
        self.d_score = {}  # Indexed by (corpus, score_title) tuples.
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

    def split_and_evaluate(self, the_model, test_size=0.3, random_state=None):
        if random_state is None:
            random_state = self.opts.random_state
        split_x_train, split_x_test, split_y_train, split_y_test = \
            train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)
        self.train_and_evaluate(the_model=the_model, x_train=split_x_train, y_train=split_y_train,
                                x_test=split_x_test, y_test=split_y_test)

    def my_k_folds(self, k=5, on_train=True, test_size: float = 0.2, random_state=None):
        if random_state is None:
            random_state = self.opts.random_state
        # splitter = GroupShuffleSplit(n_splits=k, test_size=test_size, random_state=random_state)
        splitter = GroupKFold(n_splits=k)
        if on_train:
            x = self.x_train
            y = self.y_train
            groups = self.train_group_assignments
            # groups = numpy.asarray(self.train_group_assignments)
        else:
            x = self.x
            y = self.y
            groups = self.group_assignments
            # groups = numpy.asarray(self.group_assignments)
        splits = splitter.split(X=x, y=y, groups=groups)
        for i, (train_indices, test_indices) in enumerate(splits):
            print(f"Fold {i}:")
            train_note_count = 0
            for t_i in train_indices:
                train_note_count += groups[t_i]
            test_note_count = 0
            for t_i in test_indices:
                test_note_count += groups[t_i]
            print(f"Train note count: {train_note_count}")
            print(f"Test note count: {test_note_count}")
            # print(f"  Train: index={train_indices}, group={groups[train_indices]}")
            # print(f"  Test:  index={test_indices}, group={groups[test_indices]}")
        return splits

    def tune_parameters(self, the_model):
        train_splits = self.my_k_folds(k=5, on_train=True, test_size=0.2)
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')  # , labels=labels)
        gs = GridSearchCV(estimator=the_model, param_grid=self.opts.param_grid, scoring=f1_scorer,
                          cv=train_splits, verbose=1, n_jobs=-1)
        gs.fit(self.x_train, self.y_train)
        print('best params:', gs.best_params_)
        print('best CV score:', gs.best_score_)
        print('model size: {:0.2f}M'.format(gs.best_estimator_.size_ / 1000000))

    def evaluate(self, the_model, is_trained):
        # main_split = self.my_k_folds(k=1, on_train=False, test_size=self.opts.holdout_size)
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
            ### self.my_match_rates(predictions=predictions)
            self.my_simplified_match_rates(predictions=predictions)
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

    def phrase2features(self, notes: list, staff, d_score=None):
        notes_data = DNotesData(notes=notes, staff=staff, d_score=d_score, threshold_ms=self.consonance_threshold)
        feature_list = []
        for i in range(len(notes)):
            features = self.note_func(notes_data, i, staff)
            feature_list.append(features)
        if self.reverse:
            feature_list.reverse()
        return feature_list

    def phrase2attrs(self, notes, staff, d_score=None):
        return self.phrase2features(notes, staff, d_score=d_score)

    def phrase2labels(self, handed_strike_digits: list):
        if self.reverse:
            handed_strike_digits.reverse()
        return handed_strike_digits

    def init_train_test(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.y_test_y_index = {}
        self.test_note_count = 0
        self.train_note_count = 0
        self.train_group_assignments = []

    def append_example(self, example: AnnotatedExample, is_train=False, is_test=False):
        ordered_notes = example.notes
        d_score = example.d_score
        seg_key = example.seg_key
        staff = seg_key.staff
        score_key = seg_key.score_key
        hsd_seg = example.hsd_seg
        example_key = example.example_key

        self.train_note_count = 0
        self.test_note_count = 0

        # (corpus_name, score_title, annot_index) = example_key
        # score_key = (corpus_name, score_title)
        if score_key not in self.group_id:
            new_id = len(self.group_id)
            self.group_id[score_key] = new_id
            self.group_score_key[new_id] = score_key
            self.group_note_count[new_id] = 0
            self.d_score[score_key] = d_score
        example_index = len(self.y)
        group_id = self.group_id[score_key]
        note_len = len(ordered_notes)
        self.group_note_count[group_id] += note_len
        self.annotated_note_count += note_len
        x_features = self.phrase2features(ordered_notes, staff, d_score=d_score)
        y_labels = self.phrase2labels(hsd_seg)
        self.x.append(x_features)
        self.y.append(y_labels)
        self.y_staff[example_index] = staff
        self.y_score_key[example_index] = score_key
        self.group_assignments.append(group_id)
        if example_key not in self.example_indices:
            self.example_indices[example_key] = []
        self.example_indices[example_key].append(self.good_annot_count)
        self.example_key[self.good_annot_count] = example_key
        if is_test:
            y_test_index = len(self.y_test)
            if example_key not in self.test_indices:
                self.test_indices[example_key] = []
            self.test_indices[example_key].append(self.good_annot_count)
            self.test_key[self.good_annot_count] = example_key
            self.x_test.append(x_features)
            self.y_test.append(y_labels)
            self.y_test_y_index[y_test_index] = example_index
            self.test_note_count += len(y_labels)
        elif is_train:
            self.x_train.append(x_features)
            self.y_train.append(y_labels)
            self.train_group_assignments.append(group_id)
            self.train_note_count += len(y_labels)
        self.good_annot_count += 1

    def my_train_test_split(self):
        self.init_train_test()
        groups = self.group_assignments
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.opts.holdout_size,
                                     random_state=self.opts.random_state)
        splits = splitter.split(X=self.x, y=self.y, groups=groups)
        for i, (train_indices, test_indices) in enumerate(splits):
            for t_i in train_indices:
                self.x_train.append(self.x[t_i])
                self.y_train.append(self.y[t_i])
                example_key: ExampleKey = self.example_key[t_i]
                score_key: ScoreKey = example_key.score_key()

                # (corpus_name, score_title, annot_index) = example_key
                group_id = self.group_id[score_key]
                self.train_group_assignments.append(group_id)
                note_count = len(self.y[t_i])
                self.train_note_count += note_count
            for t_i in test_indices:
                y_test_index = len(self.y_test)
                self.y_test_y_index[y_test_index] = t_i
                self.x_test.append(self.x[t_i])
                self.y_test.append(self.y[t_i])
                test_key = self.example_key[t_i]
                if test_key not in self.test_indices:
                    self.test_indices[test_key] = []
                self.test_indices[test_key].append(t_i)
                note_count = len(self.y[t_i])
                self.test_note_count += note_count

        print(f"\nTrain pieces looking less than random:\n{train_indices}")
        print(f"\nTest pieces looking less than random:\n{test_indices}")
        print(f"\nTrain note count: {self.train_note_count}")
        print(f"Test note count: {self.test_note_count}")

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
            test_d_score = self.d_score[score_title]
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

    def my_simplified_match_rates(self, predictions):
        # We need to know the associated staff and DScore for each prediction/y_test pair.
        # We identify the DScore uniquely as a (corpus_name, score_title) tuple in self.y_score_key
        # when examples are appended to self.y.
        #
        # IMPORTANT: If we are segmenting into phrases, we CANNOT produce combined scores, as these reflect
        # counts of unified slices of notes across both staves. We don't have that if the staves are divided
        # into independent phrases.
        rate_set = {
            'upper': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
            'lower': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
            'combined': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
        }
        count_set = {
            'note': {'upper': 0, 'lower': 0, 'combined': 0},
            'match': {'upper': 0, 'lower': 0, 'combined': 0}
        }
        total_counts = copy.deepcopy(count_set)
        prediction_example_count = len(predictions)
        y_test_example_count = len(self.y_test)
        if prediction_example_count != y_test_example_count:
            raise Exception("Prediction example counts do not match y_test")

        score_splits_are_valid = True
        total_score_counts = dict()
        score_ground_truth_y_indices = dict()
        for i in range(prediction_example_count):
            y_index = self.y_test_y_index[i]
            staff = self.y_staff[y_index]
            score_key = self.y_score_key[y_index]
            note_count = len(predictions[i])
            if score_key not in total_score_counts:
                total_score_counts[score_key] = copy.deepcopy(count_set)
                score_ground_truth_y_indices[score_key] = list()
            score_ground_truth_y_indices[score_key].append(y_index)  # staffs are interleaved upper, lower, upper,...
            if score_splits_are_valid:
                try:
                    match_count = DExperiment.match_count(predicted=predictions[i], ground_truth=self.y_test[i])
                    total_counts['match'][staff] += match_count
                    total_counts['note'][staff] += note_count
                    total_score_counts[score_key]['match'][staff] += match_count
                    total_score_counts[score_key]['note'][staff] += note_count

                    total_counts['match']['combined'] += match_count
                    total_counts['note']['combined'] += note_count
                    total_score_counts[score_key]['match']['combined'] += match_count
                    total_score_counts[score_key]['note']['combined'] += note_count
                except Exception as e:
                    print("Upper/lower split metrics are invalid: " + str(e))
                    splits_are_valid = False
                    score_splits_are_valid = False
        sums_of_rates = copy.deepcopy(rate_set)
        weighted_sums_of_rates = copy.deepcopy(rate_set)
        m_rates = copy.deepcopy(sums_of_rates)
        weighted_m_rates = copy.deepcopy(rate_set)
        combined_smr = total_counts['match']['combined'] / total_counts['note']['combined']
        m_rates['combined']['simple'] = combined_smr
        # The simple match rate is implicitly weighted, as it is just total matches over total notes.
        weighted_m_rates['combined']['simple'] = combined_smr

        score_match_rates = dict()
        for score_key in score_ground_truth_y_indices:
            if score_key not in score_match_rates:
                score_match_rates[score_key] = copy.deepcopy(rate_set)
            for staff in ['upper', 'lower', 'combined']:
                total_score_note_count = total_score_counts[score_key]['note'][staff]
                total_score_match_count = total_score_counts[score_key]['match'][staff]
                try:
                    if not self.opts.segmenting or not staff == 'combined':
                        score_match_rates[score_key][staff]['gen'] = total_score_match_count / total_score_note_count
                except Exception as e:
                    print("What the? " + str(e))

            # score_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['combined'],
            #                                                       score_test_fingerings['combined'])
            # base_title_match_rates[base_title]['combined']['soft'] = score_soft_match_count / combined_note_count
            # score_high_match_count = DExperiment.high_match_count(score_pred_fingering['combined'],
            #                                                       score_test_fingerings['combined'])
            # base_title_match_rates[base_title]['combined']['high'] = score_high_match_count / combined_note_count
        print("Huh?")

    def my_match_rates(self, predictions):
        # predictions contain predictions for all items in the test set.
        # We need to pull the first "exemplar" for each score and compare
        # it to every ground truth we have for the score.
        last_base_title = ''
        test_keys_for_base_title = dict()
        prediction_key_for_base_title = dict()
        splits_are_valid = True
        reggie = r'\-\d+$'
        for test_key in self.test_indices:
            (corpus_name, score_title, annot_index) = test_key
            # base_title, annot_id = str(score_title).split('-')
            base_title = re.sub(reggie, '', score_title)
            if base_title != last_base_title:
                prediction_key_for_base_title[base_title] = test_key
                test_keys_for_base_title[base_title] = list()
                # We only get to predict once per piece in Nakamura evaluation.
                last_base_title = base_title
            test_keys_for_base_title[base_title].append(test_key)
        total_counts = {
            'note': {'upper': 0, 'lower': 0, 'combined': 0},
            'match': {'upper': 0, 'lower': 0, 'combined': 0}
        }
        base_title_match_rates = dict()
        base_title_note_counts = dict()
        for base_title in prediction_key_for_base_title:
            score_splits_are_valid = True
            total_score_counts = {
                'note': {'upper': 0, 'lower': 0, 'combined': 0},
                'match': {'upper': 0, 'lower': 0, 'combined': 0}
            }
            pred_key = prediction_key_for_base_title[base_title]
            (pred_upper_i, pred_lower_i) = self.test_indices[pred_key]
            # FIXME: This is wrong. predictions line up with y_test one-to-one.
            # This is pointing to some arbitrary positions in predictions.
            # This will only work if the test examples are all lined up at the
            # beginning of the total set of examples, as they are in the
            # PIG dataset. Any random splitting of this data set breaks this
            # whole thing. We need a simpler method for when the "exemplar" situation
            # is not in play. Yeesh.
            pred_upper = predictions[pred_upper_i]
            pred_lower = predictions[pred_lower_i]
            pred_combined = pred_upper + pred_lower
            upper_note_count = len(pred_upper)
            lower_note_count = len(pred_lower)
            combined_note_count = len(pred_combined)
            if base_title not in base_title_match_rates:
                base_title_match_rates[base_title] = {
                    'upper': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
                    'lower': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
                    'combined': {'gen': 0.0, 'high': 0.0, 'soft': 0.0}
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
            score_test_fingerings = {
                'upper': list(),
                'lower': list(),
                'combined': list()
            }
            for test_key in test_keys_for_base_title[base_title]:
                # print("Compare {} to {}".format(pred_key, test_key))
                (test_upper_i, test_lower_i) = self.test_indices[test_key]
                test_upper = self.y_test[test_upper_i]
                test_lower = self.y_test[test_lower_i]
                test_combined = test_upper + test_lower
                score_test_fingerings['upper'].append(test_upper)
                score_test_fingerings['lower'].append(test_lower)
                score_test_fingerings['combined'].append(test_combined)
                if score_splits_are_valid:
                    try:
                        upper_match_count = DExperiment.match_count(predicted=pred_upper, ground_truth=test_upper)
                        total_counts['match']['upper'] += upper_match_count
                        total_counts['note']['upper'] += upper_note_count
                        total_score_counts['match']['upper'] += upper_match_count
                        total_score_counts['note']['upper'] += upper_note_count
                        lower_match_count = DExperiment.match_count(predicted=pred_lower, ground_truth=test_lower)
                        total_counts['match']['lower'] += lower_match_count
                        total_counts['note']['lower'] += lower_note_count
                        total_score_counts['match']['lower'] += lower_match_count
                        total_score_counts['note']['lower'] += lower_note_count
                    except Exception:
                        print("Upper/lower split metrics are invalid.")
                        splits_are_valid = False
                        score_splits_are_valid = False

                combined_match_count = DExperiment.match_count(predicted=pred_combined, ground_truth=test_combined)
                total_counts['match']['combined'] += combined_match_count
                total_counts['note']['combined'] += combined_note_count
                total_score_counts['match']['combined'] += combined_match_count
                total_score_counts['note']['combined'] += combined_note_count

            base_title_match_rates[base_title]['combined']['gen'] = \
                total_score_counts['match']['combined'] / total_score_counts['note']['combined']
            score_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['combined'],
                                                                  score_test_fingerings['combined'])
            base_title_match_rates[base_title]['combined']['soft'] = score_soft_match_count / combined_note_count
            score_high_match_count = DExperiment.high_match_count(score_pred_fingering['combined'],
                                                                  score_test_fingerings['combined'])
            base_title_match_rates[base_title]['combined']['high'] = score_high_match_count / combined_note_count

            if score_splits_are_valid:
                base_title_match_rates[base_title]['upper']['gen'] = \
                    total_score_counts['match']['upper'] / total_score_counts['note']['upper']
                score_upper_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['upper'],
                                                                            score_test_fingerings['upper'])
                base_title_match_rates[base_title]['upper']['soft'] = score_upper_soft_match_count / upper_note_count
                score_upper_high_match_count = DExperiment.high_match_count(score_pred_fingering['upper'],
                                                                            score_test_fingerings['upper'])
                base_title_match_rates[base_title]['upper']['high'] = score_upper_high_match_count / upper_note_count

                base_title_match_rates[base_title]['lower']['gen'] = \
                    total_score_counts['match']['lower'] / total_score_counts['note']['lower']
                score_lower_soft_match_count = DExperiment.soft_match_count(score_pred_fingering['lower'],
                                                                            score_test_fingerings['lower'])
                base_title_match_rates[base_title]['lower']['soft'] = score_lower_soft_match_count / lower_note_count
                score_lower_high_match_count = DExperiment.high_match_count(score_pred_fingering['lower'],
                                                                            score_test_fingerings['lower'])
                base_title_match_rates[base_title]['lower']['high'] = score_lower_high_match_count / lower_note_count
            else:
                for staff in ['upper', 'lower']:
                    for method in ['gen', 'high', 'soft']:
                        base_title_match_rates[base_title][staff][method] = None

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
                if not splits_are_valid and staff in ('upper', 'lower'):
                    continue
                for method in ('gen', 'high', 'soft'):
                    sums_of_rates[staff][method] += rates[staff][method]
                    weighted_sums_of_rates[staff][method] += rates[staff][method] * base_title_note_counts[base_title][staff]
                total_weights[staff] += base_title_note_counts[base_title][staff]

        for staff in ('upper', 'lower', 'combined'):
            for method in ('gen', 'high', 'soft'):
                if not splits_are_valid and staff in ('upper', 'lower'):
                    continue
                m_rates[staff][method] = sums_of_rates[staff][method] / title_count
                weighted_m_rates[staff][method] = weighted_sums_of_rates[staff][method] / total_weights[staff]

        combined_smr = total_counts['match']['combined'] / total_counts['note']['combined']
        m_rates['combined']['simple'] = combined_smr
        # The simple match rate is implicitly weighted, as it is just total matches over total notes.
        weighted_m_rates['combined']['simple'] = combined_smr

        if splits_are_valid:
            upper_smr = total_counts['match']['upper'] / total_counts['note']['upper']
            lower_smr = total_counts['match']['lower'] / total_counts['note']['lower']
            m_rates['upper']['simple'] = upper_smr
            m_rates['lower']['simple'] = lower_smr
            weighted_m_rates['upper']['simple'] = upper_smr
            weighted_m_rates['lower']['simple'] = lower_smr
        else:
            for staff in ['upper', 'lower']:
                 m_rates[staff]['simple'] = None
                 weighted_m_rates[staff]['simple'] = None

        print("\nM metrics per Nakamura:")
        pprint.pprint(m_rates)
        print("Weighted M metrics:")
        pprint.pprint(weighted_m_rates)

        return m_rates, weighted_m_rates, base_title_match_rates

    def load_data(self, clean_list):
        creal = Corporeal()
        experiment_name = self.experiment_name()
        # random.seed(27)
        example_index = 0
        for corpus_name in self.corpus_names:
            da_corpus = c.unpickle_it(obj_type="DCorpus", clean_list=clean_list,
                                      file_name=corpus_name, use_dill=True)
            if da_corpus is None:
                da_corpus = creal.get_corpus(corpus_name=corpus_name)
                if self.pickling:
                    c.pickle_it(obj=da_corpus, obj_type="DCorpus", file_name=corpus_name, use_dill=True)
            # d_scores = da_corpus.d_score_list()
            # d_random_scores = random.sample(d_scores, len(d_scores))
            for da_score in da_corpus.d_score_list():
                abcdh = da_score.abcd_header()
                annot_count = abcdh.annotation_count()
                annot = da_score.annotation_by_index(index=0)
                if self.opts.segmenting:
                    segger = ManualDSegmenter(level='.', d_annotation=annot)
                else:
                    segger = OneDSegmenter(d_annotation=annot)
                da_score.segmenter(segger)
                da_unannotated_score = copy.deepcopy(da_score)
                score_title = da_score.title()
                score_key = ScoreKey(corpus_name, score_title)
                # if score_title == 'scales_bflat_minor_melodic':
                    # print("Hang on now.")
                # if score_title != 'Sonatina 4.1':
                # continue
                for annot_index in range(annot_count):
                    annot = da_score.annotation_by_index(annot_index)
                    authority = annot.authority()
                    for staff in self.staffs:
                        # Important staff is the inner loop, so the staff annotations are next to each other
                        # in self.y.
                        example_index += 1
                        ordered_offset_note_segments = da_score.ordered_offset_note_segments(staff=staff)
                        hsd_segments = segger.segment_annotation(annotation=annot, staff=staff)
                        seg_index = 0
                        for hsd_seg in hsd_segments:
                            ordered_notes = ordered_offset_note_segments[seg_index]
                            note_len = len(ordered_notes)
                            seg_len = len(hsd_seg)
                            segment_key = SegmentKey(score_key=score_key, staff=staff, seg_index=seg_index)
                            example = AnnotatedExample(seg_key=segment_key, notes=ordered_notes, hsd_seg=hsd_seg,
                                                       annot_index=annot_index, d_score=da_unannotated_score)
                            print(f"Processing the {staff} staff of annotation {annot_index} " +
                                  f"in score {score_title} segment {seg_index} " +
                                  f"from the {corpus_name} corpus: ex. {example_index}")
                            if note_len != seg_len:
                                print("Bad annotation by {} for score {}. Notes: {} Fingers: {}".format(
                                    authority, score_title, note_len, seg_len))
                                self.bad_annot_count += 1
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
                            if self.opts.holdout_predefined and \
                                    creal.has_preset_evaluation_defined(corpus_names=self.corpus_names):
                                if creal.is_in_test_set(title=score_title, corpus_name=corpus_name):
                                    self.append_example(example, is_test=True)
                                else:
                                    self.append_example(example, is_train=True)
                            else:
                                self.append_example(example)
                        seg_index += 1
        if self.bad_annot_count != 0:
            raise Exception(f"{self.bad_annot_count} bad annotations found.")
        if not self.y_test:
            # If there be no test examples, we create our own train and test sets.
            self.my_train_test_split()

        print("Data loaded. Clean list: {}".format(list(clean_list.keys())))
        return experiment_name
