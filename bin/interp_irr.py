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
import copy
from sklearn.metrics import cohen_kappa_score
from pydactyl.dcorpus.DCorpus import DCorpus, DScore

INTERP_DIR = '/Users/dave/tb2/didactyl/dd/corpora/clementi/interp'


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

one_agg = {
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
other_agg = copy.deepcopy(one_agg)
agg_pair_counts = {
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


for da_score in da_corpus.d_score_list():
    print("Title: {}, Total notes: {}".format(da_score.title(), da_score.note_count()))
    for ref_id in interpolated:
        one_id = interpolated[ref_id][0]
        other_id = interpolated[ref_id][1]
        one_annot = da_score.annotation_by_id(identifier=one_id)
        other_annot = da_score.annotation_by_id(identifier=other_id)
        ref_annot = da_score.annotation_by_id(identifier=ref_id)
        # print(ref_annot)
        # print(one_annot)
        # print(other_annot)

        for staff in ('upper', 'both', 'lower'):
            note_count = da_score.note_count(staff=staff)
            kappa, pair_counts = da_score.cohens_kappa(
                one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=False)
            print("{} kappa for annotation {} = {}, note count: {}".format(staff, ref_id, kappa, note_count))
            print_non_zero(pair_counts)
            if staff != 'both':
                segregated_kappa, segregated_pair_counts = da_score.cohens_kappa(
                    one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=True)
                print("{} kappa for segregated annotation {} = {}, note count: {}".format(
                    staff, ref_id, kappa, note_count))
                print_non_zero(segregated_pair_counts)

            full_data = da_score.cohens_kappa_data(
                one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=False)
            one_agg['full'][staff].extend(full_data['one'])
            other_agg['full'][staff].extend(full_data['other'])
            aggregate_pair_counts(agg_pair_counts['full'][staff], full_data['pair_counts'])
            if staff != 'both':
                segregated_data = da_score.cohens_kappa_data(
                    one_id, other_id, staff=staff, common_id=ref_id, wildcard=False, segregated=True)
                one_agg['segregated'][staff].extend(segregated_data['one'])
                other_agg['segregated'][staff].extend(segregated_data['other'])
                aggregate_pair_counts(agg_pair_counts['segregated'][staff], full_data['pair_counts'])
        print()

for typ in one_agg:
    segregated = False
    if typ == 'segregated':
        segregated = True
    for staff in one_agg[typ]:
        labels = DScore.unigram_labels(staff=staff, wildcard=False, segregated=segregated)
        kappa = cohen_kappa_score(one_agg[typ][staff], other_agg[typ][staff], labels=labels)
        note_count = total_notes(pair_counts=agg_pair_counts[typ][staff])
        print("{} kappa for {} label list = {}, note count: {}".format(
            staff, typ, kappa, note_count))
        print_non_zero(agg_pair_counts[typ][staff])
