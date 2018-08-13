#!/usr/bin/env python3
import re
import unittest
from pydactyl.dactyler.Hart import Hart
from pydactyl.dactyler.Hart import HartK
from pydactyl.dcorpus.DCorpus import DCorpus
import TestConstant


class RankTest(unittest.TestCase):
    @staticmethod
    def test_hartk():
        hart = Hart()
        d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE_SHORT)
        hart.load_corpus(d_corpus=d_corpus)
        hart_rh_advice = hart.advise(staff="upper")
        print(hart_rh_advice)

        hart_lh_advice = hart.advise(staff="lower")
        print(hart_lh_advice)

        hart_k = HartK()
        hart_k.load_corpus(d_corpus=d_corpus)
        hart_k.segment_combination_method("cost")

        suggestions, costs, hds_for_gold_index = hart_k.evaluate_strike_distances(method="hamming", staff="upper", k=20)
        min_cost = None
        for i in range(len(suggestions)):
            print("{0}:::{1}".format(costs[i], suggestions[i]))
            if suggestions[i] == hart_rh_advice:
                print("GOT IT!")
                break

        for gi in hds_for_gold_index:
            print("***" + str(gi) + "***")
            hd_count = len(hds_for_gold_index[gi])
            print("HD count for GI {0} = {1}".format(gi, hd_count))
            for hdi in range(hd_count):
                if hds_for_gold_index[gi][hdi] == 0:
                    print("GOT A ZERO HAMMING DISTANCE")

        # assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
        # assert hamming_evaluations[1] == 0, "Bad fish in Hamming barrel"


if __name__ == "__main__":
    unittest.main()  # run all tests
