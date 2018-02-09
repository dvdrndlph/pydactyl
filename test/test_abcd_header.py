#!/usr/bin/env python3

import unittest
from DCorpus.DCorpus import ABCDHeader, ABCDFAnnotation


class ABCDHeaderTest(unittest.TestCase):
    MULTI_ANNOTATION_ABCD = """% abcDidactyl v5
% abcD fingering 1: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213213231231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% These are complete fingerings, with any gaps filled in.
% abcD fingering 2: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213214341231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:27:28
% These are alternate fingerings, if specified, with gaps filled in. 
% abcD fingering 3: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213214341231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:27:28
% These are additional alternate fingerings, with gaps filled in. 
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

    FINGERING_1_UPPER = "12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321"
    FINGERING_1_TRANS_DATE = "2016-09-13 17:24:43"
    FINGERING_2_AUTHORITY = "Beringer and Dunhill"
    FINGERING_2_AUTHORITY_YEAR = "1900"
    FINGERING_3_LOWER = "54321321432132123123412312345&54321321432132123123412312345&32132143213214341231234123121"
    @staticmethod
    def test_hdr_parse():
        hdr = ABCDHeader(abcd_str=ABCDHeaderTest.MULTI_ANNOTATION_ABCD)
        annot_count = hdr.annotation_count()
        assert annot_count == 3, "Bad annotation count"


if __name__ == "__main__":
    unittest.main()  # run all tests
