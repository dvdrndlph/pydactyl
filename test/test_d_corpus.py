#!/usr/bin/env python3
__author__ = 'David Randolph'
# Copyright (c) 2018 David A. Randolph.
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
from music21 import *
import re
import unittest
import TestConstant
from DCorpus.DCorpus import DPart, DScore, DCorpus

BERINGER2_SCALE_CORPUS = TestConstant.BERINGER2_DIR + "/scales.abc"


class DCorpusTest(unittest.TestCase):
    @staticmethod
    def test_abc_2part_mono():
        d_corpus = DCorpus(corpus_path=BERINGER2_SCALE_CORPUS)
        score_count = d_corpus.score_count()
        assert score_count == 38, "Bad score count"
        d_score = d_corpus.d_score_by_title("scales_g_minor_melodic")
        assert d_score is not None, "Bad score_by_title retrieval" 
        part_count = d_score.part_count()
        assert part_count == 2, "Bad part count"
        d_part = d_score.as_d_part()
        assert d_part is not None, "Bad DPart retrieval"
        assert d_part.is_monophonic() is False, "Polyphony not detected"
        d_upper = d_score.upper_d_part()
        assert d_upper, "Bad upper DPart"
        assert d_upper.is_monophonic() is True, "Monophonic upper part not detected"
        d_lower = d_score.lower_d_part()
        assert d_lower, "Bad lower DPart"
        # lower_stream = d_lower.stream()
        # lower_stream.show('text')
        assert d_lower.is_monophonic() is True, "Monophonic lower part not detected"

    @staticmethod
    def test_abc_2part_chords():
        p01_path = TestConstant.WTC_CORPUS_DIR + '/prelude01.abc'
        p09_path = TestConstant.WTC_CORPUS_DIR + '/prelude09.abc'
        d_corpus = DCorpus(corpus_path=p01_path)
        score_count = d_corpus.score_count()
        assert score_count == 1, "Bad score count"
        d_corpus.append(corpus_path=p09_path)
        assert d_corpus.score_count() == 2, "Bad append"
        titles = d_corpus.titles()
        assert len(titles) == 2, "Bad title count"
        d_score = d_corpus.d_score_by_index(0)
        assert d_score.title() == 'Prelude 1 (BWV 846)', "Bad title retrieval"
        part_count = d_score.part_count()
        assert part_count == 2, "Bad part count"
        d_part = d_score.as_d_part()
        assert d_part is not None, "Bad DPart retrieval"
        assert d_part.is_monophonic() is False, "Polyphony not detected"
        d_upper = d_score.upper_d_part()
        assert d_upper, "Bad upper DPart"
        assert d_upper.is_monophonic() is False, "Polyphonic upper part not detected"
        d_lower = d_score.lower_d_part()
        assert d_lower, "Bad lower DPart"
        assert d_lower.is_monophonic() is False, "Polyphonic lower part in Prelude 1 not detected"
        assert d_lower.is_orderly() is False, "Lower part in Prelude 1 is not orderly"
        orderly_stream = d_lower.orderly_note_stream()
        # orderly_stream.show('text')
        orderly_d_part = DPart(music21_stream=orderly_stream)
        assert orderly_d_part.is_orderly() is True, "orderly_note_stream() or is_orderly() is broken"

        # Need to check what happens to tied notes
        d_score = d_corpus.d_score_by_index(1)
        # print(title)
        d_upper = d_score.upper_d_part()
        disorderly_stream = d_upper.stream()
        disorderly_d_part = DPart(music21_stream=disorderly_stream)
        assert disorderly_d_part.is_orderly() is False, "orderly_note_stream() or is_orderly() is broken"
        # disorderly_stream.show('text')
        orderly_stream = d_upper.orderly_note_stream()
        orderly_d_part = DPart(music21_stream=orderly_stream)
        assert orderly_d_part.is_orderly() is True, "orderly_note_stream() or is_orderly() is broken"
        # disorderly_stream.show('text')

    @staticmethod
    def test_annotated_corpus():
        da_corpus = DCorpus()
        da_corpus.append_from_db(client_id='695311d7e88d5f79b4945bf45d00cc77', selection_id='21');
        da_score = da_corpus.d_score_by_index(0);
        da_title = da_score.title()
        assert da_title == 'Prelude 2 (BWV 847)', "Bad fetch by index"
        da_score = da_corpus.d_score_by_title(da_title)
        assert da_title == da_score.title(), "Bad fetch by title"
        annotation = da_score.abcd_header()
        fingering = annotation.fingering()
        finger_re = re.compile('[12345]+')
        at_re = re.compile('@')
        assert finger_re.search(fingering), "Bad fingering"
        assert at_re.search(fingering), "Bad fingering"
        upper_fingering = annotation.upper_fingering()
        assert finger_re.search(upper_fingering), "Bad upper fingering"
        assert not at_re.search(upper_fingering), "Bad upper fingering"
        lower_fingering = annotation.lower_fingering()
        assert finger_re.search(lower_fingering), "Bad upper fingering"
        assert not at_re.search(lower_fingering), "Bad upper fingering"
        assert lower_fingering != upper_fingering, "Bad split of fingerings"
        fingering_by_index = da_score.fingering(index=0)
        fingering_by_id = da_score.fingering(id=1)
        assert fingering_by_index == fingering_by_id, "Bad DSCore::fingering"
        upper_fingering_by_index = da_score.upper_fingering(index=0)
        upper_fingering_by_id = da_score.upper_fingering(id=1)
        assert upper_fingering_by_index == upper_fingering_by_id, "Bad DSCore::upper_fingering"
        lower_fingering_by_index = da_score.lower_fingering(index=0)
        lower_fingering_by_id = da_score.lower_fingering(id=1)
        assert lower_fingering_by_index == lower_fingering_by_id, "Bad DSCore::lower_fingering"

if __name__ == "__main__":
    unittest.main()  # run all tests
