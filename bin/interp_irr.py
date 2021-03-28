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
import re
import sys
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation

INTERP_DIR = '/Users/dave/tb2/didactyl/dd/corpora/clementi/interp'
da_corpus = DCorpus()
da_corpus.append_dir(corpus_dir=INTERP_DIR)

for da_score in da_corpus.d_score_list():
    print(da_score.title())
    print(da_score.note_count())
    one_annot = da_score.annotation_by_id(identifier=14)
    other_annot = da_score.annotation_by_id(identifier=15)
    print(one_annot)
    print(other_annot)

    # kappa, pair_counts = score.cohens_kappa("14", "15")
    # print("Kappa = {}".format(kappa))
    # for pair in pair_counts:
    #     if pair_counts[pair] > 0:
    #         print("{}: {}".format(pair, pair_counts[pair]))

