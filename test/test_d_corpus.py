#!/usr/bin/env python3

import unittest
import TestConstant
from DCorpus.DCorpus import DPart, DScore, DCorpus

BERINGER2_SCALE_CORPUS = TestConstant.BERINGER2_DIR + "/scales.abc"

class DCorpusTest(unittest.TestCase):
    @staticmethod
    def test_abc():
        d_corpus = DCorpus(corpus_path=BERINGER2_SCALE_CORPUS)
        score_count = d_corpus.get_score_count()
        assert score_count == 38, "Bad score count"
        d_score = d_corpus.get_d_score_by_title("scales_g_minor_melodic")
        assert d_score is not None, "Bad score_by_title retrieval" 
        voice_count = d_score.get_voice_count()
        print("Voice count: {0}".format(voice_count))
        assert voice_count == 2, "Bad voice count"


if __name__ == "__main__":
    unittest.main()  # run all tests
