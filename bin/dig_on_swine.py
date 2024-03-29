#!/usr/bin/env python
__author__ = 'David Randolph'
# Copyright (c) 2020 David A. Randolph.
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
import sys
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
from pydactyl.dactyler.Parncutt import Parncutt
from pydactyl.dcorpus.PigInOut import PigIn, PigOut, PIG_SEGREGATED_DATASET_DIR
from pydactyl.eval.Corporeal import Corporeal, COMPLETE_LAYER_ONE_STD_PIG_DIR
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
# import scamp

# s = scamp.Session()
# print(s.get_available_midi_output_devices())
# exit(0)

# pig_eater = PigIn(standardize=True)
# pig_eater.transform()
# pig_eater = PigIn()
# pig_eater.transform()
# pig_eater = PigIn(base_dir=PIG_SEGREGATED_DATASET_DIR, start_over=True)
# pig_eater.transform()

creal = Corporeal()
da_corpus = creal.get_corpus(corpus_name='complete_layer_one')
segger = ManualDSegmenter(level='.')
da_corpus.segmenter(segmenter=segger)
fingered_segments = da_corpus.fingered_ordered_offset_notes(print_summary=True)
pig_strings = PigOut.transform_fingered_ordered_offset_note_list_set(foonls=fingered_segments)
# pprint.pprint(pig_strings)
# da_corpus = creal.get_corpus(corpus_name='complete_layer_one')
# fingered_staffs = da_corpus.fingered_ordered_offset_notes(print_summary=True)
print("Whoa")


# staff = 'upper'
# k = 5
# corpus_dir = "/Users/dave/tb2/didactyl/dd/corpora/pig/PianoFingeringDataset_v1.00/abcd/"
# d_corpus = DCorpus()
# d_corpus.append_dir(corpus_dir=corpus_dir, split_header_extension='abcd')
#
# model = Parncutt()
# model.load_corpus(d_corpus=d_corpus)
# advice = model.generate_advice(staff=staff, score_index=0, k=k)
# print(advice)
