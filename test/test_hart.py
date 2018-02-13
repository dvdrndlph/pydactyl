#!/usr/bin/env python3
import re
import unittest
from Dactyler.Hart import Hart
from DCorpus.DCorpus import DCorpus
from DCorpus.DAnnotation import DAnnotation


class HartTest(unittest.TestCase):
    A_MAJ_SCALE = """% abcDidactyl v5
% abcD fingering 1: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213213231231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% These are complete fingerings, with any gaps filled in.
% abcD fingering 2: >123412312312345431321432143212312312341231231432132143213212341231231234543132143214321@<432143213214321321321432132144321432132143213213214321321421432132143213214321321432132
% Authority: Hart et al. 
% abcDidactyl END
X:7
T:scales_a_major
C:Beringer and Dunhill
%%score { ( 1 ) | ( 2 ) }
M:7/4
K:Amaj
V:1 treble
V:2 bass octave=-1
V:1
L:1/16
A,B,CD EFGA Bcde fgag fedc BAGF EDCB,:|A,4|:
CDEF GABc defg abc'b agfe dcBA GFED:|C4|:
A,B,CD EFGA Bcde fgag fedc BAGF EDCB,:|A,4|]
V:2
L:1/16
A,B,CD EFGA Bcde fgag fedc BAGF EDCB,:|A,4|:
A,B,CD EFGA Bcde fgag fedc BAGF EDCB,:|A,4|:
CDEF GABc [K:clef=treble octave=-1] defg abc'b agfe [K:clef=bass octave=-1] dcBA GFED:|C4|]
"""

    @staticmethod
    def test_hart():
        hart = Hart()
        d_corpus = DCorpus(corpus_str=HartTest.A_MAJ_SCALE)
        hart.load_corpus(d_corpus=d_corpus)
        rh_advice = hart.advise()
        right_re = re.compile('^>\d+$')
        assert right_re.match(rh_advice), "Bad right-hand, upper-staff advice"
        lh_advice = hart.advise(staff="lower")
        left_re = re.compile('^<\d+$')
        assert left_re.match(lh_advice), "Bad left-hand, lower-staff advice"
        combo_advice = hart.advise(staff="both")
        note_count = hart.score_note_count()
        print("NOTE COUNT: {0}".format(note_count))
        clean_combo_advice = re.sub('[><&]', '',  combo_advice)
        print("TEST: " + clean_combo_advice)
        d_score = d_corpus.d_score_by_index(index=0)
        gold_fingering = d_score.fingering(index=0)
        clean_gold_fingering = re.sub('[><&]', '',  gold_fingering)
        print("GOLD: " + clean_gold_fingering)

        combo_re = re.compile('^>\d+@<\d+$')
        assert combo_re.match(combo_advice), "Bad combined advice"
        hamming_evaluations = hart.evaluate_strike_distance(method="hamming", staff="both")
        for he in hamming_evaluations:
            print(he)
        assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
        assert hamming_evaluations[0] > 0, "Undetected Hamming costs"
        assert hamming_evaluations[1] == 0, "Bad fish in barrel"
        natural_evaluations = hart.evaluate_strike_distance(method="natural", staff="both")
        for he in natural_evaluations:
            print(he)
        assert natural_evaluations[0] > 0, "Undetected Hamming costs"
        assert natural_evaluations[0] > 0, "Undetected Hamming costs"
        assert natural_evaluations[1] == 0, "Bad fish in barrel"


if __name__ == "__main__":
    unittest.main()  # run all tests
