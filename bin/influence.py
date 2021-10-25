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
import pprint

from nltk.metrics.agreement import AnnotationTask
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
from pydactyl.dcorpus.PianoFingering import PianoFingering

import matplotlib
matplotlib.use('TkAgg')


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


def pf_trigram_and_triple(old_trigram, new_pf):
    new_trigram = [None, None, None]
    new_trigram[0] = old_trigram[1]
    new_trigram[1] = old_trigram[2]
    if new_pf is not None:
        new_trigram[2] = new_pf
    else:
        new_trigram[2] = None
    nova_triple = tuple(new_trigram)
    return new_trigram, nova_triple


def trigramify(pfs):
    triples = []
    trigram = [None, None, None]
    for pf in pfs:
        trigram, triple = pf_trigram_and_triple(trigram, pf)
        triples.append(triple)
    return triples


def abcdf2pf_list(abcdf):
    annot = DAnnotation(abcdf=abcdf)
    sfs = annot.score_fingerings(staff="upper")
    pfs = []
    for sf in sfs:
        pf = PianoFingering(score_fingering=sf)
        pfs.append(pf)
    return pfs


CZERNY_PFS = dict()
CZERNY_PFS[1] = abcdf2pf_list(">3545342315453423@")
CZERNY_PFS[5] = abcdf2pf_list(">145231523152312@")
CZERNY_TRIGRAMS = dict()
CZERNY_TRIGRAMS[1] = trigramify(CZERNY_PFS[1])
CZERNY_TRIGRAMS[5] = trigramify(CZERNY_PFS[5])
CZERNY_PFS_X = dict()
CZERNY_PFS_X[1] = abcdf2pf_list(">35xx3x2x15xx3x2x@")
CZERNY_PFS_X[5] = abcdf2pf_list(">14x231523152312@")

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
for corpus_name in corpora:
    annotated_scores[corpus_name] = dict()
    corpus_trigram_labels[corpus_name] = dict()
    corpus_unigram_labels[corpus_name] = dict()
    score_index = 0
    for score in corpora[corpus_name].d_score_list():
        score_number = score_numbers[score_index]
        annotated_scores[corpus_name][score_number] = []
        score_index += 1
        trigram_labels = score.trigram_strike_annotation_data()
        corpus_trigram_labels[corpus_name][score_number] = trigram_labels
        unigram_labels = score.unigram_strike_annotation_data()
        corpus_unigram_labels[corpus_name][score_number] = unigram_labels
        for annot in score.annotations():
            annotated_score = copy.deepcopy(score)
            annotated_score.finger(staff="upper", d_annotation=annot)
            annotated_scores[corpus_name][score_number].append(annotated_score)

# print(annotated_scores['advised'])
# print(corpus_trigram_labels['advised'])
# print(corpus_unigram_labels['advised'])

results = dict()
corpus_unigrams = dict()
corpus_trigrams = dict()
czerny_data = dict()

for corpus_name in annotated_scores:
    czerny_data[corpus_name] = dict()
    corpus_unigrams[corpus_name] = dict()
    corpus_trigrams[corpus_name] = dict()
    unigram_annotation_data = []
    trigram_annotation_data = []
    trigram_label_annotation_data = []
    unigram_label_annotation_data = []
    coder_id = 1
    for score_number in annotated_scores[corpus_name]:
        corpus_unigrams[corpus_name][score_number] = dict()
        corpus_trigrams[corpus_name][score_number] = dict()
        for annotated_score in annotated_scores[corpus_name][score_number]:
            corpus_unigrams[corpus_name][score_number][coder_id] = []
            corpus_trigrams[corpus_name][score_number][coder_id] = []
            item_index = 0
            orderly_d_notes = annotated_score.orderly_d_notes(staff="upper")
            # five_gram = [None, None, None, None, None]
            trigram = [None, None, None]
            for d_note in orderly_d_notes:
                trigram, triple = trigram_and_triple(trigram, d_note)
                item_id = "{}_{}".format(score_number, item_index)
                record = [coder_id, item_id, triple]
                corpus_trigrams[corpus_name][score_number][coder_id].append(triple)
                trigram_annotation_data.append(record)
                record = [coder_id, item_id, d_note.piano_fingering()]
                unigram_annotation_data.append(record)
                corpus_unigrams[corpus_name][score_number][coder_id].append(d_note.piano_fingering())
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

    for score_number in corpus_trigram_labels[corpus_name]:
        score_trigram_data = corpus_trigram_labels[corpus_name][score_number]
        score_unigram_data = corpus_unigram_labels[corpus_name][score_number]
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

    for score_id in (1, 5):
        czerny_data[corpus_name][score_id] = dict()
        score_len = len(CZERNY_PFS[score_id])
        czerny_data[corpus_name][score_id]['unigram_match'] = 0
        czerny_data[corpus_name][score_id]['unigram_proxy'] = 0
        annotator_count = 0
        total_distance = {
            'unigram': 0,
            'adjlong': 0,
            'trigram': 0,
            'nuanced': 0
        }

        # for annotator_id, annotator_unigrams in corpus_unigrams[corpus_name][score_id].items():
        for annotator_id in corpus_unigrams[corpus_name][score_id]:
            annotator_unigrams = corpus_unigrams[corpus_name][score_id][annotator_id]
            annotator_trigrams = corpus_trigrams[corpus_name][score_id][annotator_id]
            annotator_distance = {
                'unigram': 0,
                'adjlong': 0,
                'trigram': 0,
                'nuanced': 0
            }
            is_unigram_match = True
            is_unigram_proxy_match = True
            for note_index in range(score_len):
                czerny_finger = CZERNY_PFS[score_id][note_index]
                annotated_finger = annotator_unigrams[note_index]
                annotator_distance['unigram'] += PianoFingering.delta_hamming(czerny_finger, annotated_finger)
                annotator_distance['adjlong'] += PianoFingering.delta_adjacent_long(czerny_finger, annotated_finger)
                czerny_trigram = CZERNY_TRIGRAMS[score_id][note_index]
                annotated_trigram = annotator_trigrams[note_index]
                annotator_distance['trigram'] += PianoFingering.tau_trigram(czerny_trigram, annotated_trigram)
                annotator_distance['nuanced'] += PianoFingering.tau_nuanced(czerny_trigram, annotated_trigram)
                if not PianoFingering.is_unigram_match(annotated_finger, czerny_finger):
                    is_unigram_match = False
                    if not PianoFingering.is_adjacent_long(annotated_finger, czerny_finger):
                        is_unigram_proxy_match = False
            if is_unigram_match:
                czerny_data[corpus_name][score_id]['unigram_match'] += 1
            if is_unigram_proxy_match:
                czerny_data[corpus_name][score_id]['unigram_proxy'] += 1
            total_distance['unigram'] += annotator_distance['unigram']
            total_distance['adjlong'] += annotator_distance['adjlong']
            total_distance['trigram'] += annotator_distance['trigram']
            total_distance['nuanced'] += annotator_distance['nuanced']
            annotator_count += 1
        czerny_data[corpus_name][score_id]['annotator_count'] = annotator_count
        czerny_data[corpus_name][score_id]['score_length'] = score_len
        czerny_data[corpus_name][score_id]['total_distance'] = total_distance

    annot_task = AnnotationTask(data=unigram_annotation_data, distance=PianoFingering.delta_hamming)
    results[('hamming_unigram_object', corpus_name)] = annot_task.alpha()
    annot_task = AnnotationTask(data=unigram_annotation_data, distance=PianoFingering.delta_adjacent_long)
    results[('adjlong_unigram_object', corpus_name)] = annot_task.alpha()

    annot_task = AnnotationTask(data=trigram_annotation_data, distance=PianoFingering.tau_trigram)
    results[('hamming_trigram_object', corpus_name)] = annot_task.alpha()
    annot_task = AnnotationTask(data=trigram_annotation_data, distance=PianoFingering.tau_nuanced)
    results[('nuanced_trigram_object', corpus_name)] = annot_task.alpha()

    # annot_task = AnnotationTask(data=unigram_label_annotation_data)
    # results[('unigram_label', corpus_name)] = annot_task.alpha()
    # annot_task = AnnotationTask(data=trigram_label_annotation_data)
    # results[('trigram_label', corpus_name)] = annot_task.alpha()
    # These sanity checks pass: They match their object counterparts.

# print(trigram_label_annotation_data)
# print(trigram_annotation_data)
# print(unigram_label_annotation_data)
# print(unigram_annotation_data)

for (distance_function, corpus_name) in sorted(results):
    print("{} alpha for {}: {}".format(
        distance_function.rjust(len("adjlong_unigram_object")),
        corpus_name.rjust(len("independent")),
        round(results[(distance_function, corpus_name)], 5)))

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(czerny_data)

print("Basta")

