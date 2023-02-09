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

import numpy
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

RATE_SET = {
    'upper': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
    'lower': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
    'combined': {'gen': 0.0, 'high': 0.0, 'soft': 0.0},
}
COUNT_SET = {
    'note': {'upper': 0, 'lower': 0, 'combined': 0},
    'match': {'upper': 0, 'lower': 0, 'combined': 0}
}
STAFF_COUNT_SET = {'upper': 0, 'lower': 0, 'combined': 0}
METHOD_SET = {'gen': 0.0, 'high': 0.0, 'soft': 0.0}


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

    def __lt__(self, other):
        if self.corpus_name >= other.corpus_name:
            return False
        if self.score_title >= other.score_title:
            return False
        return True


def staff_cmp(one, other):
    if one == other:
        return 0
    if one == 'upper':
        return -1  # One (upper) comes before the other (lower).
    return 1  # One (lower) comes after other (upper).


@dataclass(frozen=True)
class SegmentKey:
    staff: str
    score_key: ScoreKey
    segment_index: int = 0

    def __eq__(self, other):
        if not isinstance(other, SegmentKey):
            raise Exception("SegmentKey can only be compared to another SegmentKey.")
        if (self.score_key == other.score_key and
                self.staff == other.staff and self.segment_index == other.segment_index):
            return True
        return False

    def __lt__(self, other):
        if self.score_key < other.score_key:
            return True
        if not self.score_key == other.score_key:
            return False
        if staff_cmp(self.staff, other.staff) >= 0:
            return False
        if self.segment_index >= other.segment_index:
            return False
        return True

    def score_key(self):
        return self.score_key


class Segment:
    segment_key: SegmentKey
    pred_fingering = []
    test_fingerings = []

    def __init__(self, segment_key, pred_fingering, test_fingerings):
        self.segment_key = segment_key
        self.pred_fingering = pred_fingering
        self.test_fingerings = test_fingerings

    def set_prediction(self, pred_fingering):
        self.pred_fingering = pred_fingering

    def add_test_fingering(self, test_fingering):
        self.test_fingerings.append(test_fingering)


@dataclass(frozen=True)
class ExampleKey:
    segment_key: SegmentKey
    annot_index: int = 0

    def __eq__(self, other):
        if not isinstance(other, ExampleKey):
            raise Exception("ExampleKey can only be compared to another ExampleKey.")
        if self.segment_key == other.segment_key and self.annot_index == other.annot_index:
            return True
        return False

    def score_key(self):
        return self.segment_key.score_key


class AnnotatedExample:
    def __init__(self, segment_key: SegmentKey, notes, hsd_seg, annot_index, d_score):
        self.example_key = ExampleKey(segment_key=segment_key, annot_index=annot_index)
        self.hsd_seg = hsd_seg
        self.notes = notes
        self.segment_key = segment_key
        self.annot_index = annot_index
        self.d_score = d_score

    def score_key(self):
        return self.segment_key.score_key()


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

        self.y_example_key = {}
        self.y_index_for_example_key = {}
        self.y_segment_key = {} # Map example index to associated SegmentKey (corpus name, DScore title, staff, seg index)
        self.y_score_key = {}  # Map example index to associated ScoreKey (corpus name, score title).
        self.y_test_y_index = {}  # Map the test example index back to its index in y.

        # Keep track of groups for k-fold cross-validation.
        self.group_id = {}  # Map info tuples to integer group identifiers.
        self.group_note_count = {}  # Map group identifiers to counts of notes contained therein.
        self.group_key = {}  # Map group identifiers to counts of notes contained therein.
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

    def train(self, the_model, x_train=None, y_train=None):
        if y_train is None:
            x_train = self.x_train
            y_train = self.y_train
        the_model.fit(x_train, y_train)

    def train_and_evaluate(self, the_model, x_train=None, y_train=None, x_test=None, y_test=None):
        self.train(the_model, x_train=x_train, y_train=y_train)
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
        splitter = GroupKFold(n_splits=k)
        # splitter = GroupShuffleSplit(n_splits=k, test_size=test_size, random_state=random_state)
        # splitter = GroupShuffleSplit(n_splits=k, random_state=random_state)
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
        # test_group_map = dict()
        # train_group_map = dict()
        # for i, (train_indices, test_indices) in enumerate(splits):
        #     print(f"Fold {i}:")
        #     train_note_count = 0
        #     for t_i in train_indices:
        #         note_count = len(self.y[t_i])
        #         train_note_count += note_count
        #         train_group_map[t_i] = groups[t_i]
        #     test_note_count = 0
        #     for t_i in test_indices:
        #         note_count = len(self.y[t_i])
        #         test_note_count += note_count
        #         test_group_map[t_i] = groups[t_i]
        #     print(f"Train note count: {train_note_count}")
        #     print(f"Test note count: {test_note_count}")
        #     print("Train example index to group index mapping:")
        #     print(train_group_map)
        #     print("Test example index to group index mapping:")
        #     print(test_group_map)
        #     # print(f"  Train:\nindex={train_indices},\ngroup={groups[train_indices]}")
        #     # print(f"  Test:\nindex={test_indices},\ngroup={groups[test_indices]}")
        return splits

    def update_fold_counts(self, counts, cohort, y_index):
        staff = self.y_staff[y_index]
        note_count = len(self.y[y_index])
        counts[cohort]['example']['combined'] += 1
        counts[cohort]['note']['combined'] += note_count
        counts[cohort]['example'][staff] += 1
        counts[cohort]['note'][staff] += note_count

    def train_and_evaluate_folds(self, the_model, on_train=True, output_results=False):
        splits = self.my_k_folds(on_train=on_train)

        y_test_y_index = dict()
        fold_results = list()
        for i, (train_indices, test_indices) in enumerate(splits):
            x_train = list()
            y_train = list()
            y_test = list()
            x_test = list()
            counts = {
                'train': {
                    'example': copy.deepcopy(STAFF_COUNT_SET),
                    'note': copy.deepcopy(STAFF_COUNT_SET)
                },
                'test': {
                    'example': copy.deepcopy(STAFF_COUNT_SET),
                    'note': copy.deepcopy(STAFF_COUNT_SET)
                },
            }
            for t_i in train_indices:
                self.update_fold_counts(counts, cohort='train', y_index=t_i)
                x_train.append(self.x[t_i])
                y_train.append(self.y[t_i])
            for t_i in test_indices:
                self.update_fold_counts(counts, cohort='test', y_index=t_i)
                x_test.append(self.x[t_i])
                test_example_index = len(y_test)
                y_test_y_index[test_example_index] = t_i
                y_test.append(self.y[t_i])

            self.train(the_model, x_train=x_train, y_train=y_train)
            if output_results:
                print("===================================================================================")
                print(f"Fold {i}")
                print("===================================================================================")
            predictions = the_model.predict(x_test)
            labels = list(the_model.classes_)
            flat_weighted_f1 = metrics.flat_f1_score(y_test, predictions, average='weighted', labels=labels)
            flat_accuracy = metrics.flat_accuracy_score(y_test, predictions)
            m_rates, weighted_m_rates, seg_match_rates = self.my_match_rates(predictions, y_test=y_test,
                                                                             y_test_y_index=y_test_y_index)
            fold_result = {
                'flat_weighted_f1': flat_weighted_f1,
                'flat_accuracy': flat_accuracy,
                'm_rates': m_rates,
                'weighted_m_rates': weighted_m_rates,
                # 'seg_match_rates': seg_match_rates,
                'counts': counts
            }
            if output_results:
                pprint.pp(fold_result)
            fold_results.append(fold_result)
        return fold_results

    def summarize_fold_results(self, results):
        precision = 4  # decimal places
        header_str = "Data Set & Segments & Annotations & Accuracy & F1 & Mgen & Mhigh & Msoft & WMgen & WMhigh & WMsoft\n"
        set_name = self.corpora_name()
        seg_count = results[0]['counts']['train']['example']['combined'] + \
                    results[0]['counts']['test']['example']['combined']
        annot_count = results[0]['counts']['train']['note']['combined'] + \
                      results[0]['counts']['test']['note']['combined']
        accuracy = 0.0
        f1 = 0.0
        m = copy.deepcopy(METHOD_SET)
        wm = copy.deepcopy(METHOD_SET)
        for result in results:
            test_annot_count = result['counts']['test']['note']['combined']
            proportion = test_annot_count / annot_count
            accuracy += result['flat_accuracy'] * proportion
            f1 += result['flat_weighted_f1'] * proportion
            for method in ('gen', 'high', 'soft'):
                m[method] += result['m_rates']['combined'][method] * proportion
                wm[method] += result['weighted_m_rates']['combined'][method] * proportion

        accuracy = round(accuracy, precision)
        f1 = round(f1, precision)
        m_gen = round(m['gen'], precision)
        m_high = round(m['high'], precision)
        m_soft = round(m['soft'], precision)
        wm_gen = round(wm['gen'], precision)
        wm_high = round(wm['high'], precision)
        wm_soft = round(wm['soft'], precision)
        data_str = f"{set_name} & {seg_count} & {annot_count} & {accuracy} & {f1} & {m_gen} & {m_high} & {m_soft} & {wm_gen} & {wm_high} & {wm_soft} \\\\"
        print(header_str + data_str)

    def summarize_more_fold_results(self, results):
        precision = 4  # decimal places
        nombre = self.corpora_name()
        seg_counts = copy.deepcopy(STAFF_COUNT_SET)
        annot_counts = copy.deepcopy(STAFF_COUNT_SET)
        accuracy = 0.0
        f1 = 0.0
        m = copy.deepcopy(RATE_SET)
        wm = copy.deepcopy(RATE_SET)
        proportion_total = 0.0
        for result in results:
            proportion = 0.0
            for staff in ('combined', 'upper', 'lower'):
                seg_counts[staff] = result['counts']['train']['example'][staff] + \
                                    result['counts']['test']['example'][staff]
                annot_counts[staff] = result['counts']['train']['note'][staff] + \
                                      result['counts']['test']['note'][staff]

                test_annot_count = result['counts']['test']['note'][staff]
                proportion = test_annot_count / annot_counts[staff]
                if staff == 'combined':
                    accuracy += result['flat_accuracy'] * proportion
                    f1 += result['flat_weighted_f1'] * proportion
                for method in ('gen', 'high', 'soft'):
                    m[staff][method] += result['m_rates'][staff][method] * proportion
                    wm[staff][method] += result['weighted_m_rates'][staff][method] * proportion
            proportion_total += proportion
        print("")
        print(f"Proportion total: {proportion_total}")
        print("")
        print(self.opts)
        print("")
        nombre_map = {
            'complete_layer_one': 'Layer One',
            'scales': 'Scale',
            'arpeggios': 'Arpeggio',
            'broken': 'Broken Chord',
            'scales-arpeggios-broken': 'Beringer',
            'complete_layer_one-scales-arpeggios-broken': 'Didactyl',
            'pig': 'PIG',
            'pig_training': 'PIG Training',
            'complete_layer_one-scales-arpeggios-broken-pig_training': 'All',
        }

        header_str = "Data Set & Segments & Annotations & Comb Accuracy & Comb F1 & Mgen & Mhigh & Msoft & WMgen & WMhigh & WMsoft"
        print(header_str)
        for staff in ('combined', 'upper', 'lower'):
            set_name = nombre_map[nombre]
            accuracy = round(accuracy, precision)
            f1 = round(f1, precision)
            m_gen = round(m[staff]['gen'], precision)
            m_high = round(m[staff]['high'], precision)
            m_soft = round(m[staff]['soft'], precision)
            wm_gen = round(wm[staff]['gen'], precision)
            wm_high = round(wm[staff]['high'], precision)
            wm_soft = round(wm[staff]['soft'], precision)
            seg_count = seg_counts[staff]
            annot_count = annot_counts[staff]
            if staff == 'combined':
                data_str = f"{set_name} & {seg_count} & {annot_count} & {accuracy} & {f1} & {m_gen} & {m_high} & {m_soft} & {wm_gen} & {wm_high} & {wm_soft} \\\\"
                print(data_str)
                header_str = "\nData Set & Staff & Segments & Annotations & Mgen & Mhigh & Msoft & WMgen & WMhigh & WMsoft"
                print(header_str)
            else:
                data_str = f"{set_name} & {staff} & {seg_count} & {annot_count} & {m_gen} & {m_high} & {m_soft} & {wm_gen} & {wm_high} & {wm_soft} \\\\"
                print(data_str)

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
            self.my_match_rates(predictions=predictions, output_results=True)
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
        example_key = example.example_key
        score_key = example_key.score_key()
        segment_key = example.segment_key
        staff = segment_key.staff
        hsd_seg = example.hsd_seg

        # self.opts.segmenting indicates that examples should be based on phrases
        # self.opts.group_by segment indicates that data splits exclude data from common segments
        # in both the training and test sets. Here we are worried about assigning a grouping key
        # for data splits. Note that if one is not segmenting, grouping by segments makes no sense
        # and should raise errors.
        if self.opts.group_by == 'segment':
            group_key = segment_key
        else:
            group_key = score_key
        self.train_note_count = 0
        self.test_note_count = 0

        # (corpus_name, score_title, annot_index) = example_key
        # segment_key = (corpus_name, score_title, segment_index)
        if group_key not in self.group_id:
            new_id = len(self.group_id)
            self.group_id[group_key] = new_id
            self.group_key[new_id] = group_key
            self.group_note_count[new_id] = 0
            self.d_score[group_key] = d_score
        example_index = len(self.y)
        group_id = self.group_id[group_key]
        note_len = len(ordered_notes)
        self.group_note_count[group_id] += note_len
        self.annotated_note_count += note_len
        x_features = self.phrase2features(ordered_notes, staff, d_score=d_score)
        y_labels = self.phrase2labels(hsd_seg)
        self.x.append(x_features)
        self.y.append(y_labels)
        self.y_staff[example_index] = staff
        self.y_segment_key[example_index] = segment_key
        self.y_score_key[example_index] = score_key
        self.group_assignments.append(group_id)
        if example_key not in self.example_indices:
            self.example_indices[example_key] = []
        self.example_indices[example_key].append(self.good_annot_count)
        self.y_example_key[self.good_annot_count] = example_key
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
        train_indices = list()
        test_indices = list()
        for i, (train_indices, test_indices) in enumerate(splits):
            for t_i in train_indices:
                self.x_train.append(self.x[t_i])
                self.y_train.append(self.y[t_i])
                example_key: ExampleKey = self.y_example_key[t_i]
                if self.opts.group_by == 'segment':
                    group_key = example_key.segment_key
                else:
                    group_key = example_key.score_key()

                # (corpus_name, score_title, annot_index) = example_key
                group_id = self.group_id[group_key]
                self.train_group_assignments.append(group_id)
                note_count = len(self.y[t_i])
                self.train_note_count += note_count
            for t_i in test_indices:
                y_test_index = len(self.y_test)
                self.y_test_y_index[y_test_index] = t_i
                self.x_test.append(self.x[t_i])
                self.y_test.append(self.y[t_i])
                test_key = self.y_example_key[t_i]
                if test_key not in self.test_indices:
                    self.test_indices[test_key] = []
                self.test_indices[test_key].append(t_i)
                note_count = len(self.y[t_i])
                self.test_note_count += note_count

        print(f"\nTrain pieces:\n{train_indices}")
        print(f"\nTest pieces:\n{test_indices}")
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
    def gen_match_rate(predicted, ground_truths):
        pred_len = len(predicted)
        gt_len = len(ground_truths[0])
        if gt_len == 0:
            raise Exception("Phrase is zero length.")
        if gt_len != pred_len:
            raise Exception("Counts do not match.")
        sequence_len = len(predicted)
        gt_count = len(ground_truths)
        match_rate = 0
        for i in range(gt_count):
            match_count = DExperiment.match_count(predicted, ground_truths[i])
            rate_contribution = match_count / (sequence_len * gt_count)
            match_rate += rate_contribution
        return match_rate

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

    def my_match_rates(self, predictions, y_test=None, y_test_y_index=None, output_results=False):
        # We need to know the associated staff and DScore for each prediction/y_test pair.
        # We identify the DScore uniquely as a (corpus_name, score_title) tuple in self.y_segment_key
        # when examples are appended to self.y.
        #
        # IMPORTANT: If we are segmenting into phrases, we CANNOT produce combined scores, as these reflect
        # counts of unified slices of notes across both staves. We don't have that if the staves are divided
        # into independent phrases.
        if y_test is None:
            y_test = self.y_test
            y_test_y_index = self.y_test_y_index

        if y_test_y_index is None:
            raise Exception("We need a mapping of test examples back to their origins (indices) in y.")

        total_counts = copy.deepcopy(COUNT_SET)
        prediction_example_count = len(predictions)
        y_test_example_count = len(y_test)
        if prediction_example_count != y_test_example_count:
            raise Exception("Prediction example counts do not match y_test")

        seg_note_counts = dict()
        total_seg_counts = dict()
        seg_pred_fingering = dict()
        seg_test_fingerings = dict()
        for i in range(prediction_example_count):
            y_index = y_test_y_index[i]
            staff = self.y_staff[y_index]
            example_key = self.y_example_key[y_index]
            annot_index = example_key.annot_index
            if self.opts.segmenting:
                segment_key = self.y_segment_key[y_index]
            else:
                segment_key = self.y_score_key[y_index]
            note_count = len(predictions[i])
            if segment_key not in total_seg_counts:
                total_seg_counts[segment_key] = copy.deepcopy(COUNT_SET)
                seg_test_fingerings[segment_key] = dict()
                seg_pred_fingering[segment_key] = dict()
                seg_note_counts[segment_key] = note_count
            if staff not in seg_test_fingerings[segment_key]:
                seg_test_fingerings[segment_key][staff] = dict()
                seg_pred_fingering[segment_key][staff] = predictions[i]
            seg_test_fingerings[segment_key][staff][annot_index] = y_test[i]
            match_count = DExperiment.match_count(predicted=predictions[i], ground_truth=y_test[i])
            total_counts['match'][staff] += match_count
            total_counts['note'][staff] += note_count
            total_seg_counts[segment_key]['match'][staff] += match_count
            total_seg_counts[segment_key]['note'][staff] += note_count

            total_counts['match']['combined'] += match_count
            total_counts['note']['combined'] += note_count
            total_seg_counts[segment_key]['match']['combined'] += match_count
            total_seg_counts[segment_key]['note']['combined'] += note_count

        seg_match_rates = dict()
        sums_of_rates = copy.deepcopy(RATE_SET)
        weighted_sums_of_rates = copy.deepcopy(RATE_SET)
        total_weights = {'upper': 0, 'lower': 0, 'combined': 0}
        seg_counts = copy.deepcopy(total_weights)
        for segment_key in seg_test_fingerings:
            if segment_key not in seg_match_rates:
                seg_match_rates[segment_key] = copy.deepcopy(RATE_SET)
            for staff in ['upper', 'lower', 'combined']:
                if staff == 'combined':
                    if self.opts.segmenting:
                        continue
                    if 'lower' not in seg_pred_fingering[segment_key]:
                        print("Hold on there.")
                    if 'upper' not in seg_pred_fingering[segment_key]:
                        print("Hold on here now.")
                    combined_pred_fingering = seg_pred_fingering[segment_key]['upper'] + \
                                              seg_pred_fingering[segment_key]['lower']
                    combined_test_fingerings = list()
                    for test_annot_index in sorted(seg_test_fingerings[segment_key]['upper']):
                        upper_test_fingering = seg_test_fingerings[segment_key]['upper'][test_annot_index]
                        lower_test_fingering = seg_test_fingerings[segment_key]['lower'][test_annot_index]
                        combined_fingering = upper_test_fingering + lower_test_fingering
                        combined_test_fingerings.append(combined_fingering)
                    pred_fingering = combined_pred_fingering
                    note_count = len(pred_fingering)
                    test_fingerings = combined_test_fingerings
                    seg_match_rates[segment_key]['combined']['gen'] = \
                        DExperiment.gen_match_rate(combined_pred_fingering, combined_test_fingerings)
                elif staff in seg_pred_fingering[segment_key]:
                    pred_fingering = seg_pred_fingering[segment_key][staff]
                    test_fingerings = list()
                    for test_annot_index in sorted(seg_test_fingerings[segment_key][staff]):
                        test_fingerings.append(seg_test_fingerings[segment_key][staff][test_annot_index])
                    note_count = len(pred_fingering)
                else:
                    continue
                seg_counts[staff] += 1
                total_weights[staff] += note_count
                seg_match_rates[segment_key][staff]['gen'] = DExperiment.gen_match_rate(pred_fingering, test_fingerings)

                seg_soft_match_count = DExperiment.soft_match_count(pred_fingering, test_fingerings)
                seg_match_rates[segment_key][staff]['soft'] = seg_soft_match_count / note_count

                seg_high_match_count = DExperiment.high_match_count(pred_fingering, test_fingerings)
                seg_match_rates[segment_key][staff]['high'] = seg_high_match_count / note_count
                for method in ('gen', 'high', 'soft'):
                    sums_of_rates[staff][method] += seg_match_rates[segment_key][staff][method]
                    weighted_sums_of_rates[staff][method] += note_count * seg_match_rates[segment_key][staff][method]

        m_rates = copy.deepcopy(sums_of_rates)
        weighted_m_rates = copy.deepcopy(RATE_SET)
        combined_smr = total_counts['match']['combined'] / total_counts['note']['combined']
        m_rates['combined']['simple'] = combined_smr
        # The simple match rate is implicitly weighted, as it is just total matches over total notes.
        weighted_m_rates['combined']['simple'] = combined_smr
        seg_count = len(seg_pred_fingering)
        for staff, rates in sums_of_rates.items():
            for method in ('gen', 'high', 'soft'):
                if seg_count != 0 and total_weights[staff] != 0:
                    m_rates[staff][method] = sums_of_rates[staff][method] / seg_counts[staff]
                    weighted_m_rates[staff][method] = weighted_sums_of_rates[staff][method] / total_weights[staff]

        if output_results:
            pprint.pprint(seg_match_rates)
            print("\nM metrics per Nakamura:")
            pprint.pprint(m_rates)
            print("Weighted M metrics:")
            pprint.pprint(weighted_m_rates)

        return m_rates, weighted_m_rates, seg_match_rates

    def load_data(self, clean_list):
        creal = Corporeal()
        experiment_name = self.experiment_name()
        # random.seed(27)
        example_index = 0
        for corpus_name in self.corpus_names:
            da_corpus = c.unpickle_it(obj_type="DCorpus", clean_list=clean_list,
                                      file_name=corpus_name, use_dill=True)
            if da_corpus is None:
                if self.opts.randomize_corpora:
                    corpus_seed = self.opts.random_state
                else:
                    corpus_seed = None
                da_corpus = creal.get_corpus(corpus_name=corpus_name, random_state=corpus_seed)
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
                # if score_title == 'scales_bflat_major':
                #     print("Hang on now.")
                # if score_title != 'Sonatina 4.1':
                # continue
                for annot_index in range(annot_count):
                    annot = da_score.annotation_by_index(annot_index)
                    authority = annot.authority()
                    for staff in self.staffs:
                        # Important staff is the inner loop, so the staff annotations are next to each other
                        # in self.y. FIXME: This isn't true. Multiple segments can pile up under each staff.
                        # It should be true if we aren't segmenting.
                        example_index += 1
                        ordered_offset_note_segments = da_score.ordered_offset_note_segments(staff=staff)
                        hsd_segments = segger.segment_annotation(annotation=annot, staff=staff)
                        segment_index = 0
                        for hsd_seg in hsd_segments:
                            ordered_notes = ordered_offset_note_segments[segment_index]
                            note_len = len(ordered_notes)
                            seg_len = len(hsd_seg)
                            segment_key = SegmentKey(score_key=score_key, staff=staff, segment_index=segment_index)
                            example = AnnotatedExample(segment_key=segment_key, notes=ordered_notes, hsd_seg=hsd_seg,
                                                       annot_index=annot_index, d_score=da_unannotated_score)
                            print(f"Processing the {staff} staff of annotation {annot_index} " +
                                  f"in score {score_title} segment {segment_index} " +
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
                            segment_index += 1
        if self.bad_annot_count != 0:
            raise Exception(f"{self.bad_annot_count} bad annotations found.")
        if not self.y_test:
            # If there be no test examples, we create our own train and test sets.
            self.my_train_test_split()

        print("Data loaded. Clean list: {}".format(list(clean_list.keys())))
        return experiment_name
