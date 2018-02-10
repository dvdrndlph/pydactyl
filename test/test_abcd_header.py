#!/usr/bin/env python3

import unittest
import pprint
from DCorpus.DCorpus import ABCDHeader, ABCDFAnnotation


class ABCDHeaderTest(unittest.TestCase):
    MULTI_ANNOTATION_ABCD = """% abcDidactyl v5
% abcD fingering 1: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213213231231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% These are complete fingerings, with any gaps filled in.
% abcD fingering 2: 123<12341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@>543213214321321231234>12312345&54321321432132123123412312345&32132143213214341231234123121
% Authority:  Beringer and Dunhill (1492)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:27:28
% These are alternate fingerings, if specified, with gaps filled in. 
% abcD fingering 3: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213214341231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:27:28
% These are additional alternate fingerings, with gaps filled in. 
% This is another comment line.
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
    FINGERING_1_COMMENTS = """These are complete fingerings, with any gaps filled in."""
    FINGERING_2_AUTHORITY = "Beringer and Dunhill"
    FINGERING_2_AUTHORITY_YEAR = "1492"
    FINGERING_3_LOWER = "54321321432132123123412312345&54321321432132123123412312345&32132143213214341231234123121"
    FINGERING_3_COMMENTS = """These are additional alternate fingerings, with gaps filled in. 
This is another comment line."""

    @staticmethod
    def test_hdr_parse():
        hdr = ABCDHeader(abcd_str=ABCDHeaderTest.MULTI_ANNOTATION_ABCD)
        annot_count = hdr.annotation_count()
        assert annot_count == 3, "Bad annotation count"
        f1_upper = hdr.upper_fingering(identifier="1")
        assert f1_upper == ABCDHeaderTest.FINGERING_1_UPPER, "Bad upper fingering for annotation 1 by id"
        f1_upper = hdr.upper_fingering(index=0)
        assert f1_upper == ABCDHeaderTest.FINGERING_1_UPPER, "Bad upper fingering for annotation 1 by index"
        f1_annot = hdr.annotation_by_id(identifier=1)
        assert f1_annot.comments() == ABCDHeaderTest.FINGERING_1_COMMENTS, "Bad comments"
        f1_trans_date = f1_annot.transcription_date()
        assert f1_trans_date == ABCDHeaderTest.FINGERING_1_TRANS_DATE, "Bad transcription date"
        f2_annot = hdr.annotation_by_id(identifier=2)
        f2_authority = f2_annot.authority()
        assert f2_authority == ABCDHeaderTest.FINGERING_2_AUTHORITY, "Bad authority"
        f2_annot = hdr.annotation(index=1)
        f2_authority_year = f2_annot.authority_year()
        assert f2_authority_year == ABCDHeaderTest.FINGERING_2_AUTHORITY_YEAR, "Bad authority year"
        assert hdr.lower_fingering(identifier=3) == ABCDHeaderTest.FINGERING_3_LOWER, "Bad lower fingering by id"
        assert hdr.lower_fingering(index=2) == ABCDHeaderTest.FINGERING_3_LOWER, "Bad lower fingering by index"
        f3_annot = hdr.annotation(identifier=3)
        assert f3_annot.comments() == ABCDHeaderTest.FINGERING_3_COMMENTS, "Bad comments"

    @staticmethod
    def test_abcdf_annotation():
        hdr = ABCDHeader(abcd_str=ABCDHeaderTest.MULTI_ANNOTATION_ABCD)
        clean_fingering = ABCDHeaderTest.FINGERING_3_LOWER.replace('&', '')
        fingering_count = len(clean_fingering)
        f3_annot = hdr.annotation(identifier=3)
        f3_ast = f3_annot.parse()
        assert f3_ast.upper, "Failed parse"
        parsed_fingering_count = f3_annot.pedaled_fingering_count(staff="lower")
        assert parsed_fingering_count == fingering_count, "Bad lower fingering count"
        segregated_digits = f3_annot.segregated_strike_digits(staff="lower")
        assert clean_fingering == segregated_digits, "Bad segregated lower digits"
        f1_annot = hdr.annotation(identifier=1)
        clean_fingering = ABCDHeaderTest.FINGERING_1_UPPER.replace('&', '')
        segregated_digits = f1_annot.segregated_strike_digits(staff="upper")
        assert clean_fingering == segregated_digits, "Bad segregated upper digits"
        f2_annot = hdr.annotation(identifier=2)
        none_digits = f2_annot.segregated_strike_digits(staff="upper")
        assert none_digits is None, "Unsegregated digits not detected"
        opposite_digits = f2_annot.segregated_strike_digits(staff="lower", hand=">")
        lower_abcdf = f2_annot.lower_abcdf()
        clean_fingering = lower_abcdf.replace('>', '')
        clean_fingering = clean_fingering.replace('&', '')
        assert clean_fingering == opposite_digits, "Bad opposite segregated digits"


if __name__ == "__main__":
    unittest.main()  # run all tests
