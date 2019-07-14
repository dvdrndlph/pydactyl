__author__ = 'David Randolph'
# Copyright (c) 2014-2018 David A. Randolph.
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

from pprint import pprint
from music21.articulations import Fingering
from pydactyl.dcorpus.DAnnotation import DAnnotation
from pydactyl.dcorpus.DCorpus import DCorpus

HACKED_6_1_2_FINGERED = """% abcDidactyl v6
% abcD fingering 1: >23323432145122321252213551252412@
% Authority: Parncutt Model (1997)
% Transcriber: Didactyl
% Transcription date: 2019-06-12 13:49:06
% abcD fingering 2: >23x2343214512232125221355125xx13@
% Authority: Parncutt Model (1997)
% Transcriber: Didactyl
% Transcription date: 2019-06-12 13:49:06
% abcD fingering 3: >23x234321<2>512232125221355125xx13@
% Authority: Parncutt Model (1997)
% Transcriber: Didactyl
% Transcription date: 2019-06-12 13:49:06
% abcDidactyl END
X:612
T:Sonatina 6.1.2 Hacked
C:Muzio Clementi
Z:Public Domain (PianoXML typeset)
%%score { ( 1 2 ) | ( 3 4 ) }
L:1/8
M:4/4
Q:1/4=101
I:linebreak $
K:D
V:1 treble 
L:1/16
V:2 treble
V:3 bass
V:4 bass
V:1
"^Allegro con spirito."  (F2 | %954
 .G2).G2 {FG}AGFG {/f}e4z2!p! G2 |$ .F2.F2 (GFEF d6 F2) | %956
 (F2E2G2B2) (B2C2E2A2) | [EG]4 z4 [DF]4 z4|]
V:3
[K:bass] z|z8|z8|z8|z8|]
"""


class PianoFingering(Fingering):
    """
    A music21 fingering for piano with full semantics per abcDF.
    FIXME: Currently ignores "alt" fingerings.

    >>> annot = DAnnotation(abcdf=">1<2(34)3-14-2/5-1;>1p5_.@")
    >>> sf_count = annot.score_fingering_count()
    >>> sf_count
    7
    >>> sfs = annot.score_fingerings()
    Traceback (most recent call last): 
        ...
    Exception: Score fingerings ordered within upper and lower staff only.
    >>> sfs = annot.score_fingerings(staff="upper")
    >>> simple = PianoFingering(score_fingering=sfs[0])
    >>> simple.fingerNumber
    1
    
    >>> lefty = PianoFingering(score_fingering=sfs[1])
    >>> lefty.fingerNumber
    2
    >>> lefty.strike_hand()
    '<'
    >>> lefty.strike_digit()
    2
    >>> lefty.is_dampered()
    False
    >>> lefty.segment_boundary()
    >>> # pprint(sfs[2])
    >>> # strike_handed_digits = DAnnotation.strike_handed_digits_for_score_fingering(sf=sfs[2], staff="upper", hand="<")
    >>> orny = PianoFingering(score_fingering=sfs[2], hand="<")
    >>> orny.ornamented()
    True
    >>> orny.strike_hand()
    '<'
    >>> orny.strike_hand(index=1)
    '<'
    >>> orny.strike_digit(index=1)
    4
    >>> subby = PianoFingering(score_fingering=sfs[3], hand="<")
    >>> subby.strike_digit()
    3
    >>> subby.release_digit()
    1
    >>> alt = PianoFingering(score_fingering=sfs[4], hand="<")
    >>> alt.strike_digit()
    4
    >>> alt.release_digit()
    2
    >>> alt.segment_boundary()
    ';'
    >>> # pprint(sfs[6])
    >>> ped = PianoFingering(score_fingering=sfs[6], hand=">")
    >>> ped.strike_digit()
    5
    >>> ped.is_dampered()
    True
    >>> ped.is_softened()
    True
    >>> ped.segment_boundary()
    '.'


    >>> corpse = DCorpus(corpus_str=HACKED_6_1_2_FINGERED)
    >>> score = corpse.d_score_by_index(0)
    >>> abcdh = score.abcd_header()
    >>> annot1 = abcdh.annotation_by_id(1)
    >>> sf1 = annot1.score_fingering_at_index(index=0, staff="upper")
    >>> pf = PianoFingering(score_fingering=sf1)
    >>> pf.fingerNumber
    2
    >>> upper_part = score.d_part(staff="upper")
    """
    def __init__(self, score_fingering=None, staff="upper", hand=None, soft="f", damper="^"):
        self._softs = []
        self._dampers = []

        self._strike_handed_digits = []
        self._release_handed_digits = []
        finger_number = None
        if score_fingering is not None:
            self._score_fingering = score_fingering
            strike_handed_digits = DAnnotation.handed_digits_for_score_fingering(
                sf=score_fingering, staff=staff, hand=hand)
            self._strike_handed_digits = strike_handed_digits
            finger_number = DAnnotation.digit_only(strike_handed_digits[0])
            release_handed_digits = DAnnotation.handed_digits_for_score_fingering(
                sf=score_fingering, action="release", staff=staff, hand=hand)
            self._release_handed_digits = release_handed_digits
            dampers, softs = DAnnotation.pedals_for_score_fingering(
                sf=score_fingering, damper=damper, soft=soft)
            self._dampers = dampers
            self._softs = softs

        super().__init__(fingerNumber=finger_number)

    def segment_boundary(self):
        sf = self._score_fingering
        return sf.segmenter

    def is_softened(self, index=0):
        if self._softs[index] == 'p':
            return True
        return False

    def is_dampered(self, index=0):
        if self._dampers[index] == '_':
            return True
        return False

    def strike_handed_digits(self):
        return self._strike_handed_digits

    def strike_hand(self, index=0):
        shf = self._strike_handed_digits[index]
        hand = DAnnotation.digit_hand(shf) 
        return hand

    def strike_digit(self, index=0):
        """
        The integer digit striking the pitch at the specified index.
        An ornamented note (e.g. trill) will have multiple pitches/fingerings for a
        single note head.
        """
        shf = self._strike_handed_digits[index]
        digit = DAnnotation.digit_only(shf)
        return digit

    def strike_hand_and_digit(self, index=0):
        return self.strike_hand(index=index), self.strike_digit(index=index)

    def release_handed_digits(self):
        rhds = []
        index = 0
        for rhd in self._release_handed_digits:
            if rhd is None:
                rhds.append(self._strike_handed_digits[index])
            else:
                rhds.append(rhd)
            index += 1
        return rhds

    def release_hand(self, index=0):
        rhd = self._release_handed_digits[index]
        if rhd is None:
            rhd = self._strike_handed_digits[index]
        hand = DAnnotation.digit_hand(rhd)
        return hand

    def release_digit(self, index=0):
        """
        The integer digit releasing the pitch at the specified index.
        An ornamented note (e.g. trill) will have multiple pitches/fingerings for a
        single note head.
        """
        rhd = self._release_handed_digits[index]
        if rhd is None:
            rhd = self._strike_handed_digits[index]
        digit = DAnnotation.digit_only(rhd)
        return digit

    def release_hand_and_digit(self, index=0):
        return self.release_hand(index=index), self.release_digit(index=index)

    def ornamented(self):
        if len(self._strike_handed_digits) > 1:
            return True
        return False

    def last_hand_used(self):
        last_index = len(self._release_handed_digits)
        last_hand = self.release_hand(index=last_index)
        return last_hand

    def last_damper(self):
        return self._dampers[-1]

    def last_soft(self):
        return self._softs[-1]


if __name__ == "__main__":
    import doctest
    doctest.testmod()