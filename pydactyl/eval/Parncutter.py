__author__ = 'David Randolph'
# Copyright (c) 2014-2018 David A. Randolph.
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

from .DEval import DEval
from pydactyl.dactyler.Parncutt import Parncutt
from pydactyl.dactyler.Parncutt import Jacobs
from pydactyl.dactyler.Parncutt import Badgerow
from pydactyl.dcorpus.DCorpus import DCorpus
import pymysql

last_digit = {
    0: 3,
    # B is cyclic pattern
    2: 1,
    3: 2,
    4: 2,
    5: 1,
    6: 1
}


class Parncutter(DEval):
    def __init__(self):
        super().__init__()
        self._conn = pymysql.connect(host='127.0.0.1', port=3306, user='didactyl', passwd='', db='didactyl2')
        self._dactyler = Parncutt(segment_combiner="cost")

    def load_published_parncutt(self):
        query = """select exercise, abc_fragment
                     from parncutt
                    order by exercise"""
        curs = self._conn.cursor()
        curs.execute(query)

        abc_content = None
        for row in curs:
            abc_content = row[1]
            self._d_corpus.append(corpus_str=abc_content)
        self._dactyler.load_corpus(d_corpus=self._d_corpus)

        query_gold = """select exercise, fingering, subject_count
                          from parncutt_published
                         order by exercise"""
        curs.execute(query_gold)

        prior_ex_id = None
        exercise_upper_gold = dict()
        for row_gold in curs:
            ex_id = int(row_gold[0])
            gold_fingering_str = '>' + str(row_gold[1])
            subject_count = row_gold[2]
            if ex_id != prior_ex_id:
                exercise_upper_gold = dict()
                self._gold['upper'].append(exercise_upper_gold)
                prior_ex_id = ex_id

            exercise_upper_gold[gold_fingering_str] = subject_count

    def map_at_perfect_recall(self, staff="upper"):
        avg_p_sum = 0
        for i in range(7):
            if i == 1:
                result = self.score_avg_p_at_perfect_recall(score_index=i, staff="upper",
                                                            cycle=4, last_digit=None)
            else:
                result = self.score_avg_p_at_perfect_recall(score_index=i, staff="upper",
                                                            cycle=None, last_digit=last_digit[i])
            # {'relevant': tp_count, 'p_at_rank': precisions_at_rank, 'avg_p': avg_p}
            avg_p_sum += result['avg_p']
        mean_avg_p = avg_p_sum/7
        return mean_avg_p

    def p_r_at_k(self, staff="upper", k=5):
        results = list()
        for i in range(7):
            if i == 1:
                result = self.score_p_r_at_k(score_index=i, staff="upper",
                                             cycle=4, last_digit=None, k=k)
            else:
                result = self.score_p_r_at_k(score_index=i, staff="upper",
                                             cycle=None, last_digit=last_digit[i], k=k)
            results.append(result)
        return results

    def compare_jacobs(self):
        jacobs = Jacobs(segment_combiner="cost")
        jacobs.load_corpus(d_corpus=self._d_corpus)

    def compare_badgerow(self):
        justin = Badgerow(segment_combiner="cost")
        justin.load_corpus(d_corpus=self._d_corpus)

