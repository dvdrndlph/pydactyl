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
    def __init__(self, dactyler=Parncutt(segment_combiner="cost"), corpus="parncutt_published"):
        super().__init__()
        self._conn = pymysql.connect(host='127.0.0.1', port=3306, user='didactyl', passwd='', db='didactyl2')
        self._dactyler = dactyler
        self._corpus = corpus

    def load_data(self):
        query = """select exercise, abc_fragment
                     from parncutt
                    order by exercise"""
        curs = self._conn.cursor()
        curs.execute(query)

        for row in curs:
            abc_content = row[1]
            self._d_corpus.append(corpus_str=abc_content)
        self._dactyler.load_corpus(d_corpus=self._d_corpus)

        # Fetch all of the fingerings for the Parncutt corpus collected from our
        # (North) American cohort of pianists.
        query_gold = """
            select exercise,
                   parncutt_fingering as fingering,
                   total as subject_count
              from parncutt_binary"""
        if self._corpus == "american_parncutt_pure":
            # Exclude any fingerings when the pianist was provided (Czerny) advice.
            # (Include only advice that came in with no assistance and is therefore "pure."
            query_gold = """
                select exercise, parncutt as fingering, count(*) as subject_count
                  from parncutt_american_pure
                 group by exercise, parncutt"""
        if self._corpus == "parncutt_published":
            # The fingerings reported in the Parncutt paper.
            query_gold = """
                select exercise, fingering, subject_count
                  from parncutt_published
                 order by exercise"""

        curs.execute(query_gold)

        prior_ex_id = None
        exercise_upper_gold = list()
        for row_gold in curs:
            ex_id = int(row_gold[0])
            gold_fingering_str = '>' + str(row_gold[1])
            subject_count = row_gold[2]
            if ex_id != prior_ex_id:
                exercise_upper_gold = list()
                self._gold['upper'].append(exercise_upper_gold)
                prior_ex_id = ex_id
            for i in range(subject_count):
                exercise_upper_gold.append(gold_fingering_str)

    def print_abcdf_files(self, target_dir="/tmp"):
        return

    def map_at_perfect_recall(self, staff="upper"):
        avg_p_sum = 0
        details = list()
        query_results = list()
        for i in range(7):
            if i == 1:
                result = self.score_avg_p_at_perfect_recall(score_index=i, staff=staff,
                                                            cycle=4, last_digit=None)
            else:
                result = self.score_avg_p_at_perfect_recall(score_index=i, staff=staff,
                                                            cycle=None, last_digit=last_digit[i])
            details.append(result)
            # {'relevant': tp_count, 'p_at_rank': precisions_at_rank, 'avg_p': avg_p}
            avg_p_sum += result['avg_p']
            query_results.append(result['avg_p'])
        mean_avg_p = avg_p_sum/7
        return mean_avg_p, query_results, details

    def map_at_k(self, staff="upper", k=10):
        avg_p_sum = 0
        details = list()
        query_results = list()
        for i in range(7):
            if i == 1:
                result = self.score_avg_p_at_k(score_index=i, staff="upper",
                                               cycle=4, last_digit=None, k=k)
            else:
                result = self.score_avg_p_at_k(score_index=i, staff="upper",
                                               cycle=None, last_digit=last_digit[i], k=k)
            details.append(result)
            # {'relevant': tp_count, 'p_at_rank': precisions_at_rank, 'avg_p': avg_p}
            avg_p_sum += result['avg_p']
            query_results.append(result['avg_p'])
        mean_avg_p = avg_p_sum/7
        return mean_avg_p, query_results, details

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

    def dcg_at_k(self, staff="upper", base="2", k=10):
        total_dcg = 0
        results = list()
        for i in range(7):
            if i == 1:
                dcg_at_k = self.score_dcg_at_k(score_index=i, staff="upper",
                                               cycle=4, last_digit=None, base=base, k=k)
            else:
                dcg_at_k = self.score_dcg_at_k(score_index=i, staff="upper", base=base,
                                               cycle=None, last_digit=last_digit[i], k=k)
            results.append(dcg_at_k)
            # {'relevant': tp_count, 'p_at_rank': precisions_at_rank, 'avg_p': avg_p}
            total_dcg += dcg_at_k
        mean = total_dcg/7
        return mean, results, None

    def ndcg_at_k(self, staff="upper", base="2", k=10):
        total_ndcg = 0
        results = list()
        for i in range(7):
            if i == 1:
                ndcg_at_k = self.score_ndcg_at_k(score_index=i, staff="upper",
                                                 cycle=4, last_digit=None, base=base, k=k)
            else:
                ndcg_at_k = self.score_ndcg_at_k(score_index=i, staff="upper", base=base,
                                                 cycle=None, last_digit=last_digit[i], k=k)
            results.append(ndcg_at_k)
            total_ndcg += ndcg_at_k
        mean = total_ndcg/7
        return mean, results, None

    def dcpg_at_k(self, staff="upper", phi=None, p=None, k=10):
        total_dcpg = 0
        results = list()
        for i in range(7):
            if i == 1:
                dcpg_at_k = self.score_dcpg_at_k(score_index=i, staff="upper",
                                                 cycle=4, last_digit=None, phi=phi, p=p, k=k)
            else:
                dcpg_at_k = self.score_dcpg_at_k(score_index=i, staff="upper", phi=phi, p=p,
                                                 cycle=None, last_digit=last_digit[i], k=k)
            results.append(dcpg_at_k)
            total_dcpg += dcpg_at_k
        mean = total_dcpg/7
        return mean, results, None

    def ndcpg_at_k(self, staff="upper", phi=None, p=None, k=10):
        total_ndcpg = 0
        results = list()
        for i in range(7):
            if i == 1:
                ndcpg_at_k = self.score_ndcpg_at_k(score_index=i, staff="upper",
                                                   cycle=4, last_digit=None, phi=phi, p=p, k=k)
            else:
                ndcpg_at_k = self.score_ndcpg_at_k(score_index=i, staff="upper", phi=phi, p=p,
                                                   cycle=None, last_digit=last_digit[i], k=k)
            results.append(ndcpg_at_k)
            total_ndcpg += ndcpg_at_k
        mean = total_ndcpg/7
        return mean, results, None

    def err_at_k(self, staff="upper", prob_function=DEval.estimated_prob_user_happy,
                 phi=None, p=None, k=10):
        total_err = 0
        results = list()
        for i in range(7):
            if i == 1:
                err_at_k = self.score_err_at_k(score_index=i, prob_function=prob_function,
                                               staff="upper", cycle=4, last_digit=None, phi=phi, p=p, k=k)
            else:
                err_at_k = self.score_err_at_k(score_index=i, prob_function=prob_function,
                                               staff="upper", phi=phi, p=p,
                                               cycle=None, last_digit=last_digit[i], k=k)
            results.append(err_at_k)
            total_err += err_at_k
        mean = total_err/7
        return mean, results, None

    def epr_at_k(self, staff="upper", phi=None, p=None, method="hamming", k=10):
        total_epr = 0
        results = list()
        for i in range(7):
            if i == 1:
                epr_at_k = self.score_epr_at_k(score_index=i, staff=staff, method=method,
                                               cycle=4, last_digit=None, phi=phi, p=p, k=k)
            else:
                epr_at_k = self.score_epr_at_k(score_index=i, staff=staff, phi=phi, p=p, method=method,
                                               cycle=None, last_digit=last_digit[i], k=k)
            results.append(epr_at_k)
            total_epr += epr_at_k
        mean = total_epr/7
        return mean, results, None

    def wxr_at_k(self, staff="upper", phi=None, p=None, method="match", k=10):
        total_wxr = 0
        big_m = 0
        results = list()
        for i in range(7):
            if i == 1:
                wxr_at_k, c_N = self.score_wildcard_rank_at_k(score_index=i, staff=staff, method=method,
                                                              cycle=4, last_digit=None, phi=phi, p=p, k=k)
            else:
                wxr_at_k, c_N = self.score_wildcard_rank_at_k(score_index=i, staff=staff, phi=phi, p=p, method=method,
                                                              cycle=None, last_digit=last_digit[i], k=k)
            results.append(wxr_at_k)
            total_wxr += c_N * wxr_at_k
            big_m += c_N
        mean = total_wxr/big_m
        return mean, results, None


    # def compare_jacobs(self):
    #     jacobs = Jacobs(segment_combiner="cost")
    #     jacobs.load_corpus(d_corpus=self._d_corpus)
    #
    # def compare_badgerow(self):
    #     justin = Badgerow(segment_combiner="cost")
    #     justin.load_corpus(d_corpus=self._d_corpus)

