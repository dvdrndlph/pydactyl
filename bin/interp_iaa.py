#!/usr/bin/env python
__author__ = 'David Randolph'
# Copyright (c) 2020-2021 David A. Randolph.
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
import re
import copy
from sklearn.metrics import cohen_kappa_score
from pydactyl.dcorpus.DCorpus import DCorpus, DScore

INTERP_DIR = '/Users/dave/tb2/didactyl/dd/corpora/clementi/interp'


class IaaResult:
    def __init__(self, one_id, other_id, kappa, corpus_name, experiment_type='interpolate', authority_id='',
                 scope='full', staff='upper', note_count=None, pair_counts=None):
        self.one_id = one_id
        self.other_id = other_id
        self.kappa = kappa
        self.corpus_name = corpus_name
        self.experiment_type = experiment_type
        self.authority_id = authority_id
        self.scope = scope
        self.staff = staff
        self.note_count = note_count
        self.pair_counts = pair_counts

    def pair_str(self, no_hands=False):
        count_matrix = {}
        for pair in self.pair_counts:
            one, other = pair.split('_')
            if one not in count_matrix:
                count_matrix[one] = {}
            count_matrix[one][other] = self.pair_counts[pair]
        heading_str = ''
        row_strings = []
        col_num = 0
        for col in sorted(count_matrix):
            finger_str = col
            if no_hands:
                finger_str = re.sub('[<>]', '', finger_str)
            row_strings.append(finger_str)
            for row in sorted(count_matrix[col]):
                if col_num == 0:
                    finger_str = row
                    if no_hands:
                        finger_str = re.sub('[<>]', '', finger_str)
                    heading_str += ',' + finger_str
                row_strings[col_num] += ',' + str(count_matrix[row][col])
            col_num += 1
        pair_str = heading_str + "\n"
        pair_str += "\n".join(row_strings)
        return pair_str

    def annotation_count(self):
        if self.pair_counts is None:
            return 0

        total = 0
        for pair in self.pair_counts:
            total += self.pair_counts[pair]
        return total

    def __str__(self):
        annot_count = self.annotation_count()
        string = "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"".format(
            self.corpus_name, self.experiment_type, self.one_id, self.other_id, self.authority_id,
            self.scope, self.staff, self.kappa, self.note_count, annot_count)
        return string


def print_non_zero(pair_counts):
    total = 0
    for pair in pair_counts:
        if pair_counts[pair] > 0:
            print("{}:{}".format(pair, pair_counts[pair]), end=', ')
            total += pair_counts[pair]
    print("Total: {}".format(total))
    return total


def aggregate_pair_counts(aggregate, pair_counts):
    for pair in pair_counts:
        if pair not in aggregate:
            aggregate[pair] = 0
        aggregate[pair] += pair_counts[pair]


def total_notes(pair_counts):
    total = 0
    for pair in pair_counts:
        total += pair_counts[pair]
    return total


da_corpus = DCorpus()
da_corpus.append_dir(corpus_dir=INTERP_DIR)

interpolated = {
    "2": ["14", "15"],
    "3": ["16", "17"]
}

annotated = [
    "1",  # Justin
    "7",  # Anne
]

one_interp_agg = {
    'full': {
        'upper': [],
        'lower': [],
        'both': []
    },
    'segregated': {
        'upper': [],
        'lower': [],
    }
}
other_interp_agg = copy.deepcopy(one_interp_agg)
one_annot_agg = copy.deepcopy(one_interp_agg)
other_annot_agg = copy.deepcopy(one_interp_agg)

interp_agg_pair_counts = {
    'full': {
        'upper': {},
        'lower': {},
        'both': {}
    },
    'segregated': {
        'upper': {},
        'lower': {},
    }
}
annot_agg_pair_counts = copy.deepcopy(interp_agg_pair_counts)

results = []

for da_score in da_corpus.d_score_list():
    score_name = da_score.title()
    print("Title: {}, Total notes: {}".format(score_name, da_score.note_count()))
    for staff in ('upper', 'both', 'lower'):
        note_count = da_score.note_count(staff=staff)
        for ref_id in interpolated:
            one_id = interpolated[ref_id][0]  # The zeroth annotator's interpolation of advice from ref_id.
            other_id = interpolated[ref_id][1]
            # one_annot = da_score.annotation_by_id(identifier=one_id)
            # other_annot = da_score.annotation_by_id(identifier=other_id)
            # ref_annot = da_score.annotation_by_id(identifier=ref_id)
            # print(ref_annot)
            # print(one_annot)
            # print(other_annot)

            kappa, pair_counts = da_score.cohens_kappa(
                one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=False)
            result = IaaResult(one_id=one_id, other_id=other_id,
                               kappa=kappa, corpus_name=score_name, experiment_type='interpolate',
                               authority_id=ref_id, staff=staff, note_count=note_count, pair_counts=pair_counts)
            results.append(result)

            for self_index in range(len(annotated)):
                self_id = annotated[self_index]  # Annotation done from scratch.
                self_interp_id = interpolated[ref_id][self_index]
                self_kappa, self_pair_counts = da_score.cohens_kappa(
                    self_id, self_interp_id, staff=staff, common_id=ref_id, wildcard=False, segregated=False)
                results.append(IaaResult(one_id=self_id, other_id=self_interp_id,
                                         kappa=kappa, corpus_name=score_name, experiment_type='self',
                                         staff=staff, note_count=note_count, pair_counts=self_pair_counts))

            print("{} kappa for interpolation {} = {}, note count: {}".format(staff, ref_id, kappa, note_count))
            print_non_zero(pair_counts)
            if staff != 'both':
                segregated_kappa, segregated_pair_counts = da_score.cohens_kappa(
                    one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=True)
                result = IaaResult(one_id=one_id, other_id=other_id,
                                   kappa=segregated_kappa, corpus_name=score_name, experiment_type='interpolate',
                                   authority_id=ref_id, scope='segregated', staff=staff,
                                   note_count=note_count, pair_counts=pair_counts)
                results.append(result)
                print("{} kappa for segregated interpolation {} = {}, note count: {}".format(
                    staff, ref_id, segregated_kappa, note_count))
                print_non_zero(segregated_pair_counts)

            full_data = da_score.cohens_kappa_data(
                one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=False)
            one_interp_agg['full'][staff].extend(full_data['one'])
            other_interp_agg['full'][staff].extend(full_data['other'])
            aggregate_pair_counts(interp_agg_pair_counts['full'][staff], full_data['pair_counts'])
            if staff != 'both':
                segregated_data = da_score.cohens_kappa_data(
                    one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=True)
                one_interp_agg['segregated'][staff].extend(segregated_data['one'])
                other_interp_agg['segregated'][staff].extend(segregated_data['other'])
                aggregate_pair_counts(interp_agg_pair_counts['segregated'][staff], full_data['pair_counts'])
        print()

        # Now compare the annotations provided from scratch by our intrepid pianists.
        one_id = annotated[0]
        other_id = annotated[1]
        kappa, pair_counts = da_score.cohens_kappa(
            one_id, other_id, staff=staff, wildcard=False, segregated=False)
        print("{} kappa between annotations {} and {} = {}, note count: {}".format(
            staff, one_id, other_id, kappa, note_count))
        result = IaaResult(one_id=one_id, other_id=other_id, kappa=kappa,
                           corpus_name=score_name, experiment_type='annotate', staff=staff,
                           note_count=note_count, pair_counts=pair_counts)
        results.append(result)
        print_non_zero(pair_counts)

        full_data = da_score.cohens_kappa_data(
            one_id, other_id, staff=staff, wildcard=False, segregated=False)
        one_annot_agg['full'][staff].extend(full_data['one'])
        other_annot_agg['full'][staff].extend(full_data['other'])
        aggregate_pair_counts(annot_agg_pair_counts['full'][staff], full_data['pair_counts'])
        if staff != 'both':
            segregated_data = da_score.cohens_kappa_data(
                one_id, other_id, staff=staff, wildcard=False, segregated=True)
            one_annot_agg['segregated'][staff].extend(segregated_data['one'])
            other_annot_agg['segregated'][staff].extend(segregated_data['other'])
            aggregate_pair_counts(annot_agg_pair_counts['segregated'][staff], full_data['pair_counts'])

for typ in one_interp_agg:
    segregated = False
    if typ == 'segregated':
        # As it turns out, strict hand segregation does not affect results.
        # segregated = True
        continue
    for staff in ['upper', 'both', 'lower']:
        labels = DScore.unigram_labels(staff=staff, wildcard=False, segregated=segregated)
        kappa = cohen_kappa_score(one_interp_agg[typ][staff], other_interp_agg[typ][staff], labels=labels)
        note_count = total_notes(pair_counts=interp_agg_pair_counts[typ][staff])
        pair_counts = interp_agg_pair_counts[typ][staff]
        result = IaaResult(one_id='X', other_id='Z', kappa=kappa,
                           corpus_name='layer_one', experiment_type='interpolate', staff=staff,
                           note_count=note_count, pair_counts=pair_counts)
        results.append(result)
        print("{} kappa for interpolation with {} label list = {}, note count: {}".format(
            staff, typ, kappa, note_count))
        print_non_zero(interp_agg_pair_counts[typ][staff])

for typ in one_annot_agg:
    segregated = False
    if typ == 'segregated':
        # As it turns out, strict hand segregation only affects results if
        # the segregation is violated sometimes.
        # segregated = True
        continue
    for staff in ('upper', 'both', 'lower'):
        labels = DScore.unigram_labels(staff=staff, wildcard=False, segregated=segregated)
        kappa = cohen_kappa_score(one_annot_agg[typ][staff], other_annot_agg[typ][staff], labels=labels)
        note_count = total_notes(pair_counts=annot_agg_pair_counts[typ][staff])
        pair_counts = annot_agg_pair_counts[typ][staff]
        result = IaaResult(one_id='X', other_id='Z', kappa=kappa,
                           corpus_name='layer_one', experiment_type='annotate', staff=staff,
                           note_count=note_count, pair_counts=pair_counts)
        results.append(result)
        print("{} kappa for {} label list = {}, note count: {}".format(
            staff, typ, kappa, note_count))
        print_non_zero(annot_agg_pair_counts[typ][staff])

for res in results:
    print(res)