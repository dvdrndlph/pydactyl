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
from music21 import note, chord
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
 (F2E2G2B2) (B2C2E2A2) | [GE]4 z4 [FD]4 z4|]
V:3
[K:bass] z|z8|z8|z8|z8|]
"""


class PianoFingering(Fingering):
    """
    A music21 fingering for piano with full semantics per abcDF.
    FIXME: Currently ignores all "alt" fingerings.

    >>> annot = DAnnotation(abcdf=">1<2(34)3-14-2/5-1;>1p5_.xx@")
    >>> sf_count = annot.score_fingering_count()
    >>> sf_count
    9
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
    >>> x = PianoFingering(score_fingering=sfs[7], hand=">")
    >>> x.strike_digit()
    >>> x.strike_hand()
    '>'

    >>> corpse = DCorpus(corpus_str=HACKED_6_1_2_FINGERED, as_xml=True)
    >>> score = corpse.d_score_by_index(0)
    >>> PianoFingering.finger_score(d_score=score, staff="both", id=1)
    >>> upper_stream = score.stream(staff="upper")
    >>> upper_notes = list(upper_stream.flat.getElementsByClass(note.Note))

    Use the fingering() or fingerings() to pull the PianoFingering articulations
    out of a note.Note.
    >>> pf = PianoFingering.fingering(upper_notes[1])
    >>> pf.strike_digit()
    3
    >>> pf.release_hand()
    '>'
    >>> pf.release_digit()
    3
    >>> pf = PianoFingering.fingering(object=upper_notes[-1])
    >>> pf.strike_hand()
    '>'
    >>> pf.strike_digit()
    5
    >>> chords = list(upper_stream.flat.getElementsByClass(chord.Chord))
    >>> lower_fingering = PianoFingering.fingering(chords[-1], index=0)
    >>> upper_fingering = PianoFingering.fingering(chords[-1], index=1)
    >>> lower_fingering.strike_digit()
    1
    >>> upper_fingering.strike_digit()
    2
    >>> fingerings = PianoFingering.fingerings(chords[-1])
    >>> fingerings[1].strike_digit()
    2

    >>> # combo_stream = score.stream(staff="both")
    >>> # combo_stream.show()


    """
    def __init__(self, score_fingering=None, staff="upper", hand=None, soft="f", damper="^"):
        self._softs = []
        self._dampers = []

        self._strike_hands_and_digits = []
        self._release_hands_and_digits = []
        finger_number = None
        if score_fingering is not None:
            self._score_fingering = score_fingering
            hands_and_digits = DAnnotation.hands_and_digits_for_score_fingering(
                sf=score_fingering, staff=staff, hand=hand)
            self._strike_hands_and_digits = hands_and_digits['strike']
            finger_number = self._strike_hands_and_digits[0][1]
            self._release_hands_and_digits = hands_and_digits['release']
            dampers, softs = DAnnotation.pedals_for_score_fingering(
                sf=score_fingering, damper=damper, soft=soft)
            self._dampers = dampers
            self._softs = softs

        super().__init__(fingerNumber=finger_number)

    @staticmethod
    def finger_score(d_score, staff="upper", abcdh=None, d_annotation=None, id=1):
        if abcdh:
            d_annotation = abcdh.annotation_by_id(id)
        if not d_annotation:
            abcdh = d_score.abcd_header()
            d_annotation = abcdh.annotation_by_id(id)

        if staff == "both" or staff == "upper":
            PianoFingering._finger_score(d_score=d_score, d_annotation=d_annotation, staff="upper")
        if staff == "both" or staff == "lower":
            PianoFingering._finger_score(d_score=d_score, d_annotation=d_annotation, staff="lower")

    @staticmethod
    def _finger_score(d_score, d_annotation, staff="upper"):
        stream = d_score.stream(staff=staff)
        notes_and_chords = stream.flat.getElementsByClass([note.Note, chord.Chord])
        sfs = d_annotation.score_fingerings(staff=staff)
        sf_index = 0
        hand = None
        soft = 'f'
        damper = '^'
        for elem in notes_and_chords:
            if isinstance(elem, chord.Chord):
                for n in elem.notes:
                    pf = PianoFingering(score_fingering=sfs[sf_index], staff=staff,
                                        hand=hand, soft=soft, damper=damper)
                    elem.articulations.append(pf)
                    hand = pf.last_hand()
                    soft = pf.last_soft()
                    damper = pf.last_damper()
                    sf_index += 1
            else:
                pf = PianoFingering(score_fingering=sfs[sf_index], staff=staff,
                                    hand=hand, soft=soft, damper=damper)
                elem.articulations.append(pf)
                hand = pf.last_hand()
                soft = pf.last_soft()
                damper = pf.last_damper()
                sf_index += 1

    @staticmethod
    def fingerings(object):
        """
        Extract the PianoFingering articulations from the music21 
        (Note or Chord) object.
        :param object: The music21 Note or Chord
        :return: An array of PianoFingering objects as ordered in the articulations
        list.
        """
        fingerings = []
        for artic in object.articulations:
            if isinstance(artic, PianoFingering):
                fingerings.append(artic)
        return fingerings

    @staticmethod
    def fingering(object, index=0):
        """
        Extract the PianoFingering articulation from the music21
        (Note or Chord) object.
        :param object: The music21 Note or Chord
        :param index: The zero-based index of the PianoFingering to be retrieved.
        :return: A PianoFingering object.
        """
        fingerings = PianoFingering.fingerings(object=object)
        return fingerings[index]

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

    def strike_hands_and_digits(self):
        return self._strike_hands_and_digits

    def strike_hand_and_digit(self, index=0):
        return self._strike_hands_and_digits[index]

    def strike_hand(self, index=0):
        (hand, digit) = self._strike_hands_and_digits[index]
        return hand

    def strike_digit(self, index=0):
        """
        The integer digit striking the pitch at the specified index.
        An ornamented note (e.g. trill) will have multiple pitches/fingerings for a
        single note head. Returns None if no digit is assigned (wildcard or "x").
        """
        (hand, digit) = self._strike_hands_and_digits[index]
        return digit

    def release_hands_and_digits(self):
        return self._release_hands_and_digits

    def release_hand(self, index=0):
        (hand, digit) = self._release_hands_and_digits[index]
        return hand

    def release_digit(self, index=0):
        """
        The integer digit releasing the pitch at the specified index.
        An ornamented note (e.g. trill) will have multiple pitches/fingerings for a
        single note head.
        """
        (hand, digit) = self._release_hands_and_digits[index]
        return digit

    def release_hand_and_digit(self, index=0):
        return self._release_hands_and_digits[index]

    def ornamented(self):
        if len(self._strike_hands_and_digits) > 1:
            return True
        return False

    def last_hand(self):
        hand = self.release_hand(index=-1)
        return hand

    def last_damper(self):
        return self._dampers[-1]

    def last_soft(self):
        return self._softs[-1]


if __name__ == "__main__":
    import doctest
    doctest.testmod()