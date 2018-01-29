#!/usr/bin/env python3

from music21 import *
import unittest
import TestConstant
from DCorpus.DCorpus import DPart, DScore, DCorpus

BERINGER2_SCALE_CORPUS = TestConstant.BERINGER2_DIR + "/scales.abc"


class DCorpusTest(unittest.TestCase):
    @staticmethod
    def test_abc_2part_mono():
        d_corpus = DCorpus(corpus_path=BERINGER2_SCALE_CORPUS)
        score_count = d_corpus.get_score_count()
        assert score_count == 38, "Bad score count"
        d_score = d_corpus.get_d_score_by_title("scales_g_minor_melodic")
        assert d_score is not None, "Bad score_by_title retrieval" 
        part_count = d_score.get_part_count()
        assert part_count == 2, "Bad part count"
        d_part = d_score.get()
        assert d_part is not None, "Bad DPart retrieval"
        assert d_part.is_monophonic() is False, "Polyphony not detected"
        d_upper = d_score.get_upper()
        assert d_upper, "Bad upper DPart"
        assert d_upper.is_monophonic() is True, "Monophonic upper part not detected"
        d_lower = d_score.get_lower()
        assert d_lower, "Bad lower DPart"
        assert d_lower.is_monophonic() is True, "Monophonic upper part not detected"



if __name__ == "__main__":
    unittest.main()  # run all tests
