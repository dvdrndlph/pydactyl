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

from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import sys
import csv
import numpy as np
import pandas as pd
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
import matplotlib
matplotlib.use('TkAgg')

# Fingerings from people who received help from Czerny.
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

advised_corpus = DCorpus()
advised_corpus.assemble_and_append_from_db(piece_query=abc_query, fingering_query=advised_query)
indy_corpus = DCorpus()
indy_corpus.assemble_and_append_from_db(piece_query=abc_query, fingering_query=indy_query)

for adv_score in advised_corpus.d_score_list():
    trigrams = adv_score.trigram_strike_annotation_data()

print("Basta")

