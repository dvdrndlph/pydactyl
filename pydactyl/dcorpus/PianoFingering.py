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

# from pprint import pprint
from music21 import note, chord
from music21.articulations import Fingering
from pydactyl.dcorpus.DAnnotation import DAnnotation

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

    >>> from pydactyl.dcorpus.DCorpus import DCorpus
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
    >>> pf = PianoFingering.fingering(m21_object=upper_notes[-1])
    >>> pf.strike_hand()
    '>'
    >>> pf.strike_digit()
    5
    >>> bh = pf.bigram_hash()
    >>> bh['direction']
    1
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
    DELTA_EPSILON = 0.5
    TAU_EPSILON = 0.99

    CONFIDENCE_EXPLICIT = 1.0
    CONFIDENCE_CLOSED = 0.9
    CONFIDENCE_OPEN = 0.6

    ASCENDING = 1
    DESCENDING = -1
    LATERAL = 0
    HAND_CHANGE = 999

    def __init__(self, score_fingering=None, staff="upper", hand=None, soft="f", damper="^",
                 prior_sf=None, direction=0):
        self._softs = []
        self._dampers = []

        self._confidence = PianoFingering.CONFIDENCE_EXPLICIT
        self._strike_hands_and_digits = []
        self._release_hands_and_digits = []
        self._prior_hands_and_digits = []
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
            self._prior_sf = prior_sf
            hands_and_digits = DAnnotation.hands_and_digits_for_score_fingering(
                sf=prior_sf, staff=staff, hand=hand)
            if hands_and_digits:
                self._prior_hands_and_digits = hands_and_digits['release']
            else:
                self._prior_hands_and_digits = None
            self._direction = direction

        super().__init__(fingerNumber=finger_number)

    def __hash__(self):
        (strike_hand, strike_digit) = self.strike_hand_and_digit()
        (release_hand, release_digit) = self.release_hand_and_digit()
        return hash((strike_hand, strike_digit, release_hand, release_digit))

    def __repr__(self):
        (strike_hand, strike_digit) = self.strike_hand_and_digit()
        (release_hand, release_digit) = self.release_hand_and_digit()
        my_str = "{}{}-{}{}".format(strike_hand, strike_digit, release_hand, release_digit)
        return my_str

    def __eq__(self, other):
        if not isinstance(other, PianoFingering):
            return False
        (strike_hand, strike_digit) = self.strike_hand_and_digit()
        (release_hand, release_digit) = self.release_hand_and_digit()
        (other_strike_hand, other_strike_digit) = other.strike_hand_and_digit()
        (other_release_hand, other_release_digit) = other.release_hand_and_digit()
        return (strike_hand == other_strike_hand and strike_digit == other_strike_digit and
                release_hand == other_release_hand and release_digit == other_release_digit)

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
    def _pitch_direction(prior_note, current_note):
        if prior_note:
            prior_midi = prior_note.pitches[0].midi
            current_midi = current_note.pitches[0].midi
            if current_midi > prior_midi:
                return PianoFingering.ASCENDING
            elif current_midi < prior_midi:
                return PianoFingering.DESCENDING
        return PianoFingering.LATERAL

    @staticmethod
    def _finger_score(d_score, d_annotation, staff="upper"):
        stream = d_score.stream(staff=staff)
        notes_and_chords = stream.flat.getElementsByClass([note.Note, chord.Chord])
        sfs = d_annotation.score_fingerings(staff=staff)
        sf_index = 0
        hand = None
        soft = 'f'
        damper = '^'
        prior_sf = None
        prior_note = None
        direction = 0
        for elem in notes_and_chords:
            if isinstance(elem, chord.Chord):
                for n in elem.notes:
                    pf = PianoFingering(score_fingering=sfs[sf_index], staff=staff, hand=hand,
                                        soft=soft, damper=damper, prior_sf=prior_sf, direction=direction)
                    elem.articulations.append(pf)
                    hand = pf.last_hand()
                    soft = pf.last_soft()
                    damper = pf.last_damper()
                    direction = PianoFingering._pitch_direction(prior_note=prior_note, current_note=n)
                    prior_sf = sfs[sf_index]
                    prior_note = n
                    sf_index += 1
            else:
                pf = PianoFingering(score_fingering=sfs[sf_index], staff=staff, hand=hand,
                                    soft=soft, damper=damper, prior_sf=prior_sf, direction=direction)
                elem.articulations.append(pf)
                hand = pf.last_hand()
                soft = pf.last_soft()
                damper = pf.last_damper()
                direction = PianoFingering._pitch_direction(prior_note=prior_note, current_note=elem)
                prior_sf = sfs[sf_index]
                prior_note = elem
                sf_index += 1

    @staticmethod
    def fingerings(m21_object):
        """
        Extract the PianoFingering articulations from the music21 
        (Note or Chord) m21_object.
        :param m21_object: The music21 Note or Chord
        :return: An array of PianoFingering objects as ordered in the articulations
        list.
        """
        fingerings = []
        for articulation in m21_object.articulations:
            if isinstance(articulation, PianoFingering):
                fingerings.append(articulation)
        return fingerings

    @staticmethod
    def fingering(m21_object, index=0):
        """
        Extract the PianoFingering articulation from the music21
        (Note or Chord) object.
        :param m21_object: The music21 Note or Chord
        :param index: The zero-based index of the PianoFingering to be retrieved.
        :return: A PianoFingering object.
        """
        fingerings = PianoFingering.fingerings(m21_object=m21_object)
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

    def bigram_hash(self, index=0):
        """
        Elements of the "semantically enriched" bigram label for the pitch at
        the specified index. An ornamented note (e.g. trill) will have multiple
        pitches/fingerings for a single note head. Returns None if no digit
        is assigned (but rather a wildcard or "x") to either bigram element.
        """
        (strike_hand, strike_digit) = self._strike_hands_and_digits[index]
        (prior_hand, prior_digit) = self._prior_hands_and_digits[index]
        bits = {
            'from': prior_digit,
            'to': strike_digit,
            'from_hand': prior_hand,
            'to_hand': strike_hand,
            'direction': self._direction
        }
        if 'x' in (prior_digit, strike_digit):
            return None
        return bits

    @staticmethod
    def delta_hamming(one_pf, other_pf):
        if one_pf.strike_digit() != other_pf.strike_digit():
            return 1.0
        elif one_pf.strike_hand() != other_pf.strike_hand():
            return 1.0
        return 0.0

    @staticmethod
    def delta_adjacent_long(one_pf, other_pf, epsilon=None):
        if epsilon is None:
            epsilon = PianoFingering.DELTA_EPSILON

        if one_pf.strike_hand() != other_pf.strike_hand():
            return 1.0
        one_digit = one_pf.strike_digit()
        other_digit = other_pf.strike_digit()
        if one_digit == other_digit:
            return 0.0
        if (one_digit in (2, 3) and other_digit in (2, 3)) or \
                (one_digit in (3, 4) and other_digit in (3, 4)):
            return 1.0 - epsilon
        return 1.0

    @staticmethod
    def is_adjacent_long(one_pf, other_pf):
        """
        Determine if two fingerings of the input notes are interchangeable
        because they are adjacent long fingers of the same hand.
        :param one_pf: The PianoFingering of the middle note of a trigram.
        :param other_pf: Another PianoFingering for a middle note of the same sequence..
        :return: True iff they are interchangeable. False otherwise, even if
                 finger used is the same.
        """
        if one_pf is None or other_pf is None:
            return False
        if one_pf.strike_hand() != other_pf.strike_hand():
            return False
        one_digit = one_pf.strike_digit()
        other_digit = other_pf.strike_digit()
        if one_digit == other_digit:
            return False
        if (one_digit in (2, 3) and other_digit in (2, 3)) or \
                (one_digit in (3, 4) and other_digit in (3, 4)):
            return True
        return False

    @staticmethod
    def is_unigram_match(one_pf, other_pf):
        if one_pf is None and other_pf is None:
            return True
        if one_pf is None or other_pf is None:
            return False
        if one_pf.strike_hand() != other_pf.strike_hand():
            return False
        if one_pf.strike_digit() == other_pf.strike_digit():
            return True
        return False

    @staticmethod
    def is_ngram_equal(pfs, other_pfs):
        if len(pfs) != len(other_pfs):
            return False
        for i in range(0, len(pfs)):
            one_pf = pfs[i]
            other_pf = other_pfs[i]
            if not PianoFingering.is_unigram_match(one_pf, other_pf):
                return False
        return True

    @staticmethod
    def is_trigram_equal(pfs, other_pfs):
        if len(pfs) != len(other_pfs):
            return False
        return PianoFingering.is_ngram_equal(pfs, other_pfs)

    @staticmethod
    def is_trigram_similar(pfs, other_pfs, proxy_test=None):
        if proxy_test is None:
            raise Exception("No nuance without an interchange_test function.")

        if PianoFingering.is_trigram_equal(pfs, other_pfs):
            return True

        if not PianoFingering.is_unigram_match(pfs[0], other_pfs[0]):
            return False
        if not PianoFingering.is_unigram_match(pfs[2], other_pfs[2]):
            return False

        if proxy_test(pfs[1], other_pfs[1]):
            return True
        return False

    @staticmethod
    def is_similar_at(pfs, other_pfs, proxy_test=None):
        """
        Check trigram similarity around the middle index of a three-fingering sequence.
        :param pfs: An array of three PianoFingerings, with the middle fingering at index 1.
        :param other_pfs: Another sequence over the same set of notes.
        :param proxy_test: Function to determine if trigrams are indeed similar.
        :return: True iff similar. False otherwise.
        """
        if PianoFingering.is_unigram_match(pfs[1], other_pfs[1]):
            return True
        return PianoFingering.is_trigram_similar(pfs, other_pfs)

    @staticmethod
    def tau_trigram(pfs, other_pfs, proxy_test=None):
        if PianoFingering.is_trigram_equal(pfs, other_pfs):
            return 0.0
        return 1.0

    @staticmethod
    def tau_nuanced(pfs, other_pfs, proxy_test=None, epsilon=None):
        if proxy_test is None:
            proxy_test = PianoFingering.is_adjacent_long
        if epsilon is None:
            epsilon = PianoFingering.TAU_EPSILON

        note_count = len(pfs)
        check_count = len(other_pfs)
        if note_count != check_count:
            raise Exception("Note stream count mismatch")

        if PianoFingering.is_trigram_equal(pfs, other_pfs):
            return 0.0

        if PianoFingering.is_trigram_similar(pfs, other_pfs, proxy_test=proxy_test):
            return 1.0 - epsilon

        return 1.0

    @staticmethod
    def tau_relaxed(pfs, other_pfs, proxy_test=None, epsilon=None):
        """
        Though presented as a "trigram" measure, it actually takes five notes of
        context to calculate the relaxed measure.
        :param pfs: List of 5 PianoFingering objects or Nones. Note that None == None.
        :param other_pfs: Another list over the same sequence of notes.
        :param proxy_test: A function to determine if one unigram PianoFingering is
        similar to another.
        :param epsilon: The penalty to incur (0 <= epsilon <= 1) to incur if a match
        is only a proxy match.
        :return:
        """
        if proxy_test is None:
            proxy_test = PianoFingering.is_adjacent_long
        if epsilon is None:
            epsilon = PianoFingering.TAU_EPSILON

        note_count = len(pfs)
        check_count = len(other_pfs)
        if note_count != check_count:
            raise Exception("Note stream count mismatch")

        if PianoFingering.is_trigram_equal(pfs[0:3], other_pfs[0:3]):
            return 0.0

        # Okay, we need up to 4 fingerings to determine this.
        if not PianoFingering.is_similar_at(pfs[0:2], other_pfs[0:2], proxy_test=proxy_test):
            return 1.0
        if not PianoFingering.is_similar_at(pfs[1:3], other_pfs[1:3], proxy_test=proxy_test):
            return 1.0
        if not PianoFingering.is_similar_at(pfs[2:4], other_pfs[2:4], proxy_test=proxy_test):
            return 1.0

        return 1.0 - epsilon

if __name__ == "__main__":
    import doctest
    doctest.testmod()
