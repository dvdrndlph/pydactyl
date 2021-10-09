#!/usr/bin/env python
__author__ = 'David Randolph'
# Copyright (c) 2021 David A. Randolph.
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

from nltk.metrics.agreement import AnnotationTask
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
from pydactyl.dcorpus.DEvaluation import DEvalFunction

import matplotlib
matplotlib.use('TkAgg')


def unigram_distance(one, other):
    return 0.1


def trigram_distance(one, other):
    return 0.0


def trigram_and_triple(old_trigram, new_note):
    new_trigram = [None, None, None]
    new_trigram[0] = old_trigram[1]
    new_trigram[1] = old_trigram[2]
    if new_note is not None:
        new_trigram[2] = new_note.piano_fingering()
    else:
        new_trigram[2] = None
    nova_triple = tuple(new_trigram)
    return new_trigram, nova_triple


# Fingerings from the two experimental fragments.
finger_query = """
  select f.upper_staff as fingering,
         1 as weight,
         'Various Didactyl' as 'authority',
         'Pydactyl' as 'transcriber'
  from didactyl2.finger f
 inner join didactyl2.parncutt p
        on f.exercise = p.exercise
 inner join didactyl2.subject_advised sa
    on f.subject = sa.response_id
 where f.exercise = {}
   and f.upper_staff is not null
   and length(f.upper_staff) = p.length_full"""

abc_query = '''
    select exercise as piece_id,
           abc_full as abc_str
      from parncutt
      where exercise in (1, 5)
     order by exercise'''

advised_query = finger_query + " and sa.Advised = 'Yes'"
indy_query = finger_query + " and sa.Advised = 'No'"

corpora = dict()
corpora['advised'] = DCorpus()
corpora['advised'].assemble_and_append_from_db(piece_query=abc_query, fingering_query=advised_query)
corpora['independent'] = DCorpus()
corpora['independent'].assemble_and_append_from_db(piece_query=abc_query, fingering_query=indy_query)

score_numbers = [1, 5]
annotated_scores = dict()
corpus_trigram_labels = dict()
corpus_unigram_labels = dict()
for corpus in corpora:
    annotated_scores[corpus] = dict()
    corpus_trigram_labels[corpus] = dict()
    corpus_unigram_labels[corpus] = dict()
    score_index = 0
    for score in corpora[corpus].d_score_list():
        score_number = score_numbers[score_index]
        annotated_scores[corpus][score_number] = []
        score_index += 1
        trigram_labels = score.trigram_strike_annotation_data()
        corpus_trigram_labels[corpus][score_number] = trigram_labels
        unigram_labels = score.unigram_strike_annotation_data()
        corpus_unigram_labels[corpus][score_number] = unigram_labels
        for annot in score.annotations():
            annotated_score = copy.deepcopy(score)
            annotated_score.finger(staff="upper", d_annotation=annot)
            annotated_scores[corpus][score_number].append(annotated_score)

# print(annotated_scores['advised'])
# print(corpus_trigram_labels['advised'])
# print(corpus_unigram_labels['advised'])

results = dict()
for corpus in annotated_scores:
    unigram_annotation_data = []
    trigram_annotation_data = []
    trigram_label_annotation_data = []
    unigram_label_annotation_data = []
    coder_id = 1
    for score_number in annotated_scores[corpus]:
        for annotated_score in annotated_scores[corpus][score_number]:
            item_index = 0
            orderly_notes = annotated_score.orderly_d_notes(staff="upper")
            trigram = [None, None, None]
            for note in orderly_notes:
                trigram, triple = trigram_and_triple(trigram, note)
                item_id = "{}_{}".format(score_number, item_index)
                record = [coder_id, item_id, triple]
                trigram_annotation_data.append(record)
                record = [coder_id, item_id, note.piano_fingering()]
                unigram_annotation_data.append(record)
                item_index += 1
            # To include "full trigram context," we need to add two more triples.
            trigram, triple = trigram_and_triple(trigram, None)
            item_id = "{}_{}".format(score_number, item_index)
            record = [coder_id, item_id, triple]
            trigram_annotation_data.append(record)
            item_index += 1
            trigram, triple = trigram_and_triple(trigram, None)
            item_id = "{}_{}".format(score_number, item_index)
            record = [coder_id, item_id, triple]
            trigram_annotation_data.append(record)
            coder_id += 1

    for score_number in corpus_trigram_labels[corpus]:
        score_trigram_data = corpus_trigram_labels[corpus][score_number]
        score_unigram_data = corpus_unigram_labels[corpus][score_number]
        for coder_id in score_trigram_data:
            item_index = 0
            for label in score_trigram_data[coder_id]:
                item_id = "{}_{}".format(score_number, item_index)
                record = [coder_id, item_id, label]
                trigram_label_annotation_data.append(record)
                item_index += 1
            item_index = 0
            for label in score_unigram_data[coder_id]:
                item_id = "{}_{}".format(score_number, item_index)
                record = [coder_id, item_id, label]
                unigram_label_annotation_data.append(record)
                item_index += 1

    annot_task = AnnotationTask(data=unigram_annotation_data)  # distance=unigram_distance
    results[('unigram_native', corpus)] = annot_task.alpha()
    annot_task = AnnotationTask(data=trigram_annotation_data)
    results[('trigram_native', corpus)] = annot_task.alpha()

    annot_task = AnnotationTask(data=unigram_label_annotation_data)
    results[('unigram_label', corpus)] = annot_task.alpha()
    annot_task = AnnotationTask(data=trigram_label_annotation_data)
    results[('trigram_label', corpus)] = annot_task.alpha()

print(trigram_label_annotation_data)
print(trigram_annotation_data)
print(unigram_label_annotation_data)
print(unigram_annotation_data)

for (distance_function, corpus) in sorted(results):
    print("{} alpha for {}: {}".format(
        distance_function.rjust(len("trigram_nuanced")),
        corpus.rjust(len("independent")),
        round(results[(distance_function, corpus)], 5)))

# The metrics from the string labels and those from the notes should be identical.
# But they are not.

print("Basta")

