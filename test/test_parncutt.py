#!/usr/bin/env python3
import re
import unittest
from didactyl.dactyler.Parncutt import Parncutt
from didactyl.dcorpus.DCorpus import DCorpus
import TestConstant


class ParncuttTest(unittest.TestCase):
    @staticmethod
    def test_something():
        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
        parncutt.load_corpus(d_corpus=d_corpus)
        complete_rh_advice = parncutt.advise(staff="upper")
        assert complete_rh_advice, "Ahhh!"

    """
    @staticmethod
    def test_parncutt_edges():
        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.ONE_NOTE)
        parncutt.load_corpus(d_corpus=d_corpus)
        upper_rh_advice = parncutt.advise(staff="upper")
        right_re = re.compile('^>\d$')
        assert right_re.match(upper_rh_advice), "Bad one-note, right-hand, upper-staff advice"
        # both_advice = parncutt.advise(staff="both")
        # both_re = re.compile('^>\d@$')
        # assert both_re.match(both_advice), "Bad one-note, segregated, both-staff advice"

        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.ONE_BLACK_NOTE_PER_STAFF)
        parncutt.load_corpus(d_corpus=d_corpus)
        upper_advice = parncutt.advise(staff="upper")

        right_re = re.compile('^>2$')
        assert right_re.match(upper_advice), "Bad black-note, upper-staff advice"
        lower_advice = parncutt.advise(staff="lower")
        left_re = re.compile('^<2$')
        assert left_re.match(lower_advice), "Bad black-note, upper-staff advice"
        both_advice = parncutt.advise(staff="both")
        both_re = re.compile('^>2@<2$')
        assert both_re.match(both_advice), "Bad black-note, both-staff advice"
        lower_advice = parncutt.advise(staff="lower", first_digit=3)
        lower_re = re.compile('^<3$')
        assert lower_re.match(lower_advice), "Bad preset black-note, both-staff advice"

        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.TWO_WHITE_NOTES_PER_STAFF)
        parncutt.load_corpus(d_corpus=d_corpus)
        upper_advice = parncutt.advise(staff="upper")
        right_re = re.compile('^>\d\d$')
        assert right_re.match(upper_advice), "Bad two white-note, upper-staff advice"
        upper_advice = parncutt.advise(staff="upper", first_digit=3, last_digit=2)
        right_re = re.compile('^>32$')
        assert right_re.match(upper_advice), "Bad preset two white-note, upper-staff advice"

    @staticmethod
    def test_pivot_alignment():
        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
        parncutt.load_corpus(d_corpus=d_corpus)

        evaluations = parncutt.evaluate_pivot_alignment(staff="both")
        # for he in hamming_evaluations:
        # print(he)
        assert evaluations[0] > 0, "Undetected pivot alignment costs"
        assert evaluations[1] == 0, "Bad fish in pivot alignment barrel"

    @staticmethod
    def test_distance_metrics():
        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
        parncutt.load_corpus(d_corpus=d_corpus)
        complete_rh_advice = parncutt.advise(staff="upper")
        complete_rh_advice_len = len(complete_rh_advice)
        right_re = re.compile('^>\d+$')
        assert right_re.match(complete_rh_advice), "Bad right-hand, upper-staff advice"
        rh_advice = parncutt.advise(staff="upper", offset=3, first_digit=4)
        short_advice_len = len(rh_advice)
        assert complete_rh_advice_len - 3 == short_advice_len, "Bad offset for advise() call"
        ff_re = re.compile('^>4\d+$')
        assert ff_re.match(rh_advice), "Bad first finger constraint"

        rh_advice = parncutt.advise(staff="upper", offset=10, first_digit=5, last_digit=5)
        short_advice_len = len(rh_advice)
        assert complete_rh_advice_len - 10 == short_advice_len, "Bad offset for advise() call"
        ff_re = re.compile('^>5\d+5$')
        assert ff_re.match(rh_advice), "Bad first and last finger constraints"

        lh_advice = parncutt.advise(staff="lower")
        left_re = re.compile('^<\d+$')
        assert left_re.match(lh_advice), "Bad left-hand, lower-staff advice"
        combo_advice = parncutt.advise(staff="both")
        clean_combo_advice = re.sub('[><&]', '',  combo_advice)
        d_score = d_corpus.d_score_by_index(index=0)
        gold_fingering = d_score.abcdf(index=0)
        clean_gold_fingering = re.sub('[><&]', '',  gold_fingering)

        combo_re = re.compile('^>\d+@<\d+$')
        assert combo_re.match(combo_advice), "Bad combined advice"
        hamming_evaluations = parncutt.evaluate_strike_distance(method="hamming", staff="both")
        # for he in hamming_evaluations:
            # print(he)
        assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
        assert hamming_evaluations[1] == 0, "Bad fish in Hamming barrel"

        natural_evaluations = parncutt.evaluate_strike_distance(method="natural", staff="both")
        # for he in natural_evaluations:
            # print(he)
        assert natural_evaluations[0] > 0, "Undetected natural costs"
        assert natural_evaluations[1] == 0, "Bad fish in natural barrel"

        pivot_evaluations = parncutt.evaluate_strike_distance(method="pivot", staff="both")
        # for he in pivot_evaluations:
            # print(he)
        assert pivot_evaluations[0] > 0, "Undetected pivot costs"
        assert pivot_evaluations[1] == 0, "Bad fish in pivot barrel"

    @staticmethod
    def test_reentry():
        parncutt = Parncutt()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
        parncutt.load_corpus(d_corpus=d_corpus)

        reentry_hamming_evals = parncutt.evaluate_strike_reentry(method="hamming", staff="upper", gold_indices=[0, 1])
        # for rhe in reentry_hamming_evals:
            # print("RHE:{0}".format(rhe))
        assert reentry_hamming_evals[0] > 0, "Undetected Hamming reentry costs"
        assert reentry_hamming_evals[1] == 0, "Bad fish in Hamming reentry barrel"

        reentry_hamming_evals = parncutt.evaluate_strike_reentry(method="hamming", staff="both")
        # for rhe in reentry_hamming_evals:
            # print("RHE:{0}".format(rhe))
        assert reentry_hamming_evals[0] > 0, "Undetected Hamming reentry costs"
        assert reentry_hamming_evals[1] == 0, "Bad fish in Hamming reentry barrel"
        hamming_score = reentry_hamming_evals[0]

        reentry_natural_evals = parncutt.evaluate_strike_reentry(method="natural", staff="both")
        # for rne in reentry_natural_evals:
            # print("RNE:{0}".format(rne))
        assert reentry_natural_evals[0] > 0, "Undetected natural reentry costs"
        assert reentry_natural_evals[1] == 0, "Bad fish in natural reentry barrel"
        natural_score = reentry_natural_evals[0]
        assert natural_score > hamming_score, "Reentry: Natural <= Hamming"

        reentry_pivot_evals = parncutt.evaluate_strike_reentry(method="pivot", staff="both")
        # for rpe in reentry_pivot_evals:
            # print("RPE:{0}".format(rpe))
        assert reentry_pivot_evals[0] > 0, "Undetected pivot reentry costs"
        assert reentry_pivot_evals[1] == 0, "Bad fish in pivot reentry barrel"
        pivot_score = reentry_pivot_evals[0]
        assert natural_score < pivot_score, "Reentry: Natural >= Pivot"
    """

if __name__ == "__main__":
    unittest.main()  # run all tests
