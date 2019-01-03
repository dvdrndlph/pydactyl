#!/usr/bin/env python3
import re
import unittest
from pydactyl.dactyler.Parncutt import Badgerow
from pydactyl.dcorpus.DCorpus import DCorpus
import TestConstant

last_digit = {
    'A': 3,
    'B': 0,  # Cyclic pattern
    'C': 1,
    'D': 2,
    'E': 2,
    'F': 1,
    'G': 1
}

class RankTest(unittest.TestCase):
    # @staticmethod
    # def test_basic():
    #     justin = Badgerow()
    #     d_corpus = DCorpus(corpus_str=TestConstant.A_MAJ_SCALE_SHORT)
    #     justin.load_corpus(d_corpus=d_corpus)
    #     rh_advice = justin.advise(staff="upper")
    #     print(rh_advice)

    # @staticmethod
    # def test_segment_cost():
    #     justin = Badgerow(segment_combiner="cost")
    #     d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT['A'])
    #     justin.load_corpus(d_corpus=d_corpus)
    #     abcdf = ">24342313"
    #     cost, details = justin.segment_advice_cost(abcdf)
    #     print("")
    #     print(abcdf)
    #     print("Cost: {0}".format(cost))
    #     for det in details:
    #         print(det)

    @staticmethod
    def test_solutions():
        for id in last_digit:
            print("")
            print("Piece {0}".format(id))
            print("=======")
            justin = Badgerow(segment_combiner="cost")
            d_corpus = DCorpus(corpus_str=TestConstant.PARNCUTT_HUMAN_FRAGMENT[id])
            justin.load_corpus(d_corpus=d_corpus)
            if id == 'B':
                suggestions, costs, details = justin.generate_advice(staff="upper", cycle=4, k=20)
            else:
                suggestions, costs, details = justin.generate_advice(staff="upper", last_digit=last_digit[id], k=20)
            justin.report_on_advice(suggestions, costs, details)
            # for i in range(10):
            #     assert costs[i] in solutions[id],\
            #         "Missing cost {0} in {1}".format(costs[i], id)
            #     assert suggestions[i] in solutions[id][int(costs[i])],\
            #         "Missing {0} cost suggestion {1} in {2}".format(costs[i], suggestions[i], id)


if __name__ == "__main__":
    unittest.main()  # run all tests
