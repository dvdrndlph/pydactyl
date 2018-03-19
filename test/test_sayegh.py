#!/usr/bin/env python3
import re
import os.path
import unittest
from Dactyler.Sayegh import Sayegh
from DCorpus.DCorpus import DPart, DScore, DCorpus
import TestConstant


class SayeghTest(unittest.TestCase):
    PICKLE_PATH = "/tmp/sayegh.pickle"

    def __init__(self, *args, **kwargs):
        super(SayeghTest, self).__init__(*args, **kwargs)
        if not os.path.isfile(SayeghTest.PICKLE_PATH):
            sayegh = Sayegh()
            d_corpus = DCorpus()
            d_corpus.append_dir(corpus_dir=TestConstant.BERINGER2_ANNOTATED_ARPEGGIO_DIR)
            d_corpus.append_dir(corpus_dir=TestConstant.BERINGER2_ANNOTATED_SCALE_DIR)
            d_corpus.append_dir(corpus_dir=TestConstant.BERINGER2_ANNOTATED_BROKEN_CHORD_DIR)
            sayegh.train(d_corpus, annotation_indices=[0])
            sayegh.retain(pickle_path=SayeghTest.PICKLE_PATH)

    @staticmethod
    def test_training():
        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)

        # sayegh.demonstrate()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
        sayegh.load_corpus(d_corpus=d_corpus)
        upper_advice = sayegh.advise(staff="upper")
        upper_re = re.compile('^>\d+$')
        assert upper_re.match(upper_advice), "Bad upper advice"
        lower_advice = sayegh.advise(staff="lower")
        lower_re = re.compile('^<\d+$')
        assert lower_re.match(lower_advice), "Bad lower advice"
        both_advice = sayegh.advise(staff="both")
        both_re = re.compile('^>\d+@<\d+$')
        assert both_re.match(both_advice), "Bad complete advice"

    @staticmethod
    def test_sayegh_edges():
        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)
        d_corpus = DCorpus(corpus_str=TestConstant.ONE_NOTE)
        sayegh.load_corpus(d_corpus=d_corpus)
        upper_rh_advice = sayegh.advise(staff="upper")
        right_re = re.compile('^>\d$')
        assert right_re.match(upper_rh_advice), "Bad one-note, right-hand, upper-staff advice"
        both_advice = sayegh.advise(staff="both")
        both_re = re.compile('^>\d@$')
        assert both_re.match(both_advice), "Bad one-note, segregated, both-staff advice"

        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)
        d_corpus = DCorpus(corpus_str=TestConstant.ONE_BLACK_NOTE_PER_STAFF)
        sayegh.load_corpus(d_corpus=d_corpus)
        upper_advice = sayegh.advise(staff="upper")
        right_re = re.compile('^>2$')
        assert right_re.match(upper_advice), "Bad black-note, upper-staff advice"
        lower_advice = sayegh.advise(staff="lower")
        left_re = re.compile('^<2$')
        assert left_re.match(lower_advice), "Bad black-note, upper-staff advice"
        both_advice = sayegh.advise(staff="both")
        both_re = re.compile('^>2@<2$')
        assert both_re.match(both_advice), "Bad black-note, both-staff advice"
        lower_advice = sayegh.advise(staff="lower", first_digit=3)
        lower_re = re.compile('^<3$')
        assert lower_re.match(lower_advice), "Bad preset black-note, lower-staff advice"

        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)
        d_corpus = DCorpus(corpus_str=TestConstant.TWO_WHITE_NOTES_PER_STAFF)
        sayegh.load_corpus(d_corpus=d_corpus)
        upper_advice = sayegh.advise(staff="upper")
        print("Upper advice: {0}".format(upper_advice))
        right_re = re.compile('^>\d\d$')
        assert right_re.match(upper_advice), "Bad two white-note, upper-staff advice"
        upper_advice = sayegh.advise(staff="upper", first_digit=3, last_digit=2)
        right_re = re.compile('^>32$')
        assert right_re.match(upper_advice), "Bad preset two white-note, upper-staff advice"

    """
    @staticmethod
    def test_smoothing():
        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)
        sayegh.smoother(method="uniform")
        sayegh.smooth()
        training = sayegh.training()
        for staff in ('upper', 'lower'):
            for (midi1, midi2, fh1, fh2) in training[staff]:
                key = (midi1, midi2, fh1, fh2)
                assert training[staff][key] != 0, "Unsmoothed transition: {0}".format(key)
    """

    @staticmethod
    def test_distance_metrics():
        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
        sayegh.load_corpus(d_corpus=d_corpus)
        complete_rh_advice = sayegh.advise(staff="upper")
        complete_rh_advice_len = len(complete_rh_advice)
        right_re = re.compile('^>\d+$')
        assert right_re.match(complete_rh_advice), "Bad right-hand, upper-staff advice"
        rh_advice = sayegh.advise(staff="upper", first_digit=4)
        right_re = re.compile('^>4\d+$')
        assert right_re.match(rh_advice), "Bad right-hand, upper-staff advice with constrained first finger"
        rh_advice = sayegh.advise(staff="upper", first_digit=3, last_digit=1)
        right_re = re.compile('^>3\d+1$')
        assert right_re.match(rh_advice), "Bad right-hand, upper-staff advice with constrained first and last finger"
        rh_advice = sayegh.advise(staff="upper", offset=3, first_digit=4)
        short_advice_len = len(rh_advice)
        assert complete_rh_advice_len - 3 == short_advice_len, "Bad offset for advise() call"
        ff_re = re.compile('^>4\d+$')
        # print(rh_advice)
        assert ff_re.match(rh_advice), "Bad first finger constraint"

        rh_advice = sayegh.advise(staff="upper", offset=10, first_digit=5, last_digit=5)
        short_advice_len = len(rh_advice)
        assert complete_rh_advice_len - 10 == short_advice_len, "Bad offset for advise() call"
        ff_re = re.compile('^>5\d+5$')
        assert ff_re.match(rh_advice), "Bad first and last finger constraints"

        lh_advice = sayegh.advise(staff="lower")
        left_re = re.compile('^<\d+$')
        assert left_re.match(lh_advice), "Bad left-hand, lower-staff advice"
        combo_advice = sayegh.advise(staff="both")
        print("COMBO ADVICE: " + combo_advice)
        clean_combo_advice = re.sub('[><&]', '',  combo_advice)
        # print("TEST: " + clean_combo_advice)
        d_score = d_corpus.d_score_by_index(index=0)
        gold_fingering = d_score.abcdf(index=0)
        clean_gold_fingering = re.sub('[><&]', '',  gold_fingering)
        # print("GOLD: " + clean_gold_fingering)

        combo_re = re.compile('^>\d+@<\d+$')
        assert combo_re.match(combo_advice), "Bad combined advice"
        hamming_evaluations = sayegh.evaluate_strike_distance(method="hamming", staff="both")
        # for he in hamming_evaluations:
            # print(he)
        assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
        assert hamming_evaluations[2] == 0, "Bad fish in Hamming barrel"

        natural_evaluations = sayegh.evaluate_strike_distance(method="natural", staff="both")
        # for ne in natural_evaluations:
            # print(ne)
        assert natural_evaluations[0] > 0, "Undetected natural costs"
        assert natural_evaluations[2] == 0, "Bad fish in natural barrel"

        pivot_evaluations = sayegh.evaluate_strike_distance(method="pivot", staff="both")
        # for he in pivot_evaluations:
            # print(he)
        assert pivot_evaluations[0] > 0, "Undetected pivot costs"
        assert pivot_evaluations[2] == 0, "Bad fish in pivot barrel"

    @staticmethod
    def test_reentry():
        sayegh = Sayegh()
        sayegh.recall(pickle_path=SayeghTest.PICKLE_PATH)
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE)
        sayegh.load_corpus(d_corpus=d_corpus)

        reentry_hamming_evals = sayegh.evaluate_strike_reentry(method="hamming", staff="upper", gold_indices=[0, 2])
        for rhe in reentry_hamming_evals:
            print("RHE:{0}".format(rhe))
        assert reentry_hamming_evals[0] > 0, "Undetected Hamming reentry costs"
        assert reentry_hamming_evals[1] == 0, "Bad fish in Hamming reentry barrel"

        reentry_hamming_evals = sayegh.evaluate_strike_reentry(method="hamming", staff="both")
        for rhe in reentry_hamming_evals:
            print("RHE:{0}".format(rhe))
        assert reentry_hamming_evals[0] > 0, "Undetected Hamming reentry costs (both staves)"
        assert reentry_hamming_evals[2] == 0, "Bad fish in Hamming reentry barrel (both staves)"
        hamming_score = reentry_hamming_evals[0]

        reentry_natural_evals = sayegh.evaluate_strike_reentry(method="natural", staff="both")
        for rne in reentry_natural_evals:
            print("RNE:{0}".format(rne))
        assert reentry_natural_evals[0] > 0, "Undetected natural reentry costs (both staves)"
        assert reentry_natural_evals[2] == 0, "Bad fish in natural reentry barrel (both staves)"
        natural_score = reentry_natural_evals[0]
        print("{0} > {1}".format(natural_score, hamming_score))
        assert natural_score > hamming_score, "Reentry: Natural <= Hamming"

        reentry_pivot_evals = sayegh.evaluate_strike_reentry(method="pivot", staff="both")
        for rpe in reentry_pivot_evals:
            print("RPE:{0}".format(rpe))
        assert reentry_pivot_evals[0] > 0, "Undetected pivot reentry costs (both staves)"
        assert reentry_pivot_evals[2] == 0, "Bad fish in pivot reentry barrel (both staves)"
        pivot_score = reentry_pivot_evals[0]
        print("{0} > {1}".format(pivot_score, natural_score))
        assert natural_score < pivot_score, "Reentry: Natural >= Pivot"


if __name__ == "__main__":
    unittest.main()  # run all tests
