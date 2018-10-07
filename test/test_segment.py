#!/usr/bin/env python3
import re
import unittest
from pydactyl.dactyler.Parncutt import Parncutt
from pydactyl.dcorpus.DCorpus import DCorpus
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
import TestConstant


class SegmentTest(unittest.TestCase):
    def test_malody(self):
        model = Parncutt(segmenter=ManualDSegmenter(),
                         segment_combiner="cost")
        d_corpus = DCorpus(paths=["/Users/dave/malody.abcd"])
        model.load_corpus(d_corpus=d_corpus)
        advice = model.advise()
        print(advice)
        # Gold-standard embedded in input file.
        hamming_dists = model.evaluate_strike_distance()
        print(hamming_dists)

if __name__ == "__main__":
    unittest.main()  # run all tests
