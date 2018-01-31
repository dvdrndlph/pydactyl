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
        # lower_stream = d_lower.get_stream()
        # lower_stream.show('text')
        assert d_lower.is_monophonic() is True, "Monophonic lower part not detected"

    @staticmethod
    def test_abc_2part_chords():
        p01_path = TestConstant.WTC_CORPUS_DIR + '/prelude01.abc'
        p09_path = TestConstant.WTC_CORPUS_DIR + '/prelude09.abc'
        d_corpus = DCorpus(corpus_path=p01_path)
        score_count = d_corpus.get_score_count()
        assert score_count == 1, "Bad score count"
        d_corpus.append(corpus_path=p09_path)
        assert d_corpus.get_score_count() == 2, "Bad append"
        titles = d_corpus.get_titles()
        assert len(titles) == 2, "Bad title count"
        d_score = d_corpus.get_d_score_by_index(0)
        title = d_score.get_title()
        assert title == 'Prelude 1 (BWV 846)', "Bad title retrieval"
        part_count = d_score.get_part_count()
        assert part_count == 2, "Bad part count"
        d_part = d_score.get()
        assert d_part is not None, "Bad DPart retrieval"
        assert d_part.is_monophonic() is False, "Polyphony not detected"
        d_upper = d_score.get_upper()
        assert d_upper, "Bad upper DPart"
        upper_stream = d_upper.get_stream()
        upper_stream.show('text')
        assert d_upper.is_monophonic() is False, "Polyphonic upper part not detected"
        d_lower = d_score.get_lower()
        assert d_lower, "Bad lower DPart"
        assert d_lower.is_monophonic() is False, "Polyphonic lower part in Prelude 1 not detected"
        assert d_lower.is_orderly() is False, "Lower part in Prelude 1 is not orderly"
        orderly_stream = d_lower.get_orderly_note_stream()
        # orderly_stream.show('text')
        orderly_d_part = DPart(music21_stream=orderly_stream)
        assert orderly_d_part.is_orderly() is True, "get_orderly_note_stream() or is_orderly() is broken"

        # Need to check what happens to tied notes
        d_score = d_corpus.get_d_score_by_index(1)
        title = d_score.get_title()
        print(title)
        d_upper = d_score.get_upper()
        # disorderly_stream = d_upper.get_stream()
        # disorderly_stream.show('text')
        orderly_stream = d_upper.get_orderly_note_stream()
        orderly_d_part = DPart(music21_stream=orderly_stream)
        assert orderly_d_part.is_orderly() is True, "get_orderly_note_stream() or is_orderly() is broken"


if __name__ == "__main__":
    unittest.main()  # run all tests
