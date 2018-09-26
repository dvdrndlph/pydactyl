#!/usr/bin/env python3
import re
import unittest
from pydactyl.dactyler.Parncutt import Parncutt
from pydactyl.dcorpus.DCorpus import DCorpus
from pydactyl.dcorpus.ManualDSegmenter import ManualDSegmenter
import TestConstant


class SegmentTest(unittest.TestCase):
    def test_four_note_example(self):
        parncutt = Parncutt(segmenter=ManualDSegmenter(), segment_combiner="cost")
        # parncutt.segment_combiner(method="cost")
        d_corpus = DCorpus(corpus_str=TestConstant.FOUR_NOTES)
        parncutt.load_corpus(d_corpus=d_corpus)
        suggestions, costs, details = parncutt.generate_advice(staff="upper", k=2)
        self.assertEqual(len(suggestions), 2, "No loops in that dog in top ten")
        parncutt.report_on_advice(suggestions, costs, details)


if __name__ == "__main__":
    unittest.main()  # run all tests
