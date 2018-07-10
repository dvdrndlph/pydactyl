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

REPO_ROOT = "/Users/dave/tb2"
CORPORA_DIR = REPO_ROOT + "/didactyl/dd/corpora"
BERINGER2_DIR = CORPORA_DIR + "/beringer2"
BERINGER2_ARPEGGIO_CORPUS = BERINGER2_DIR + "/arpeggios.abc"
BERINGER2_ANNOTATED_ARPEGGIO_DIR = BERINGER2_DIR + "/arpeggios"
BERINGER2_ANNOTATED_BROKEN_CHORD_DIR = BERINGER2_DIR + "/broken_chords"
BERINGER2_ANNOTATED_SCALE_DIR = BERINGER2_DIR + "/scales"

WTC_CORPUS_DIR = REPO_ROOT + '/dvdrndlph.github.io/didactyl/wtc'

# Right hand only, sans annotation.
BERINGER_DIR = CORPORA_DIR + "/beringer"
BERINGER_BC_CORPUS = BERINGER_DIR + "/broken_chords.abc"


ONE_NOTE = """% abcDidactyl v5
% abcD fingering 1: 1@
% Authority: Foo Bar (1968)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% abcD fingering 2: >2@
% Authority: Bar Baz (1066)
% abcDidactyl END
X:1
T:One Note
C:John Doe
M:4/4
K:C
V:1 treble
L:1/4
c4|]"""

TWO_NOTES = """% abcDidactyl v5
% abcD fingering 1: 12@
% Authority: Foo Bar (1968)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% abcD fingering 2: >23@
% Authority: Bar Baz (1066)
% abcDidactyl END
X:2
T:Dos Notes
C:John Doe
M:4/4
K:C
V:1 treble
L:1/4
b2c2|]"""

ONE_BLACK_NOTE_PER_STAFF = """% abcDidactyl v5
% abcD fingering 1: 12@21
% Authority: Foo Bar (1968)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% abcD fingering 2: >23@<32
% Authority: Bar Baz (1066)
% abcDidactyl END
X:1
T:One Note Each
C:John Doe
%%score { ( 1 ) | ( 2 ) }
M:4/4
K:C
V:1 treble
V:2 bass octave=-1
V:1 treble
L:1/4
_b4|]
V:2
L:1/4
_E4|]"""

TWO_WHITE_NOTES_PER_STAFF = """% abcDidactyl v5
% abcD fingering 1: 12@21
% Authority: Foo Bar (1968)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% abcD fingering 2: >23@<32
% Authority: Bar Baz (1066)
% abcDidactyl END
X:1
T:One Note
C:John Doe
%%score { ( 1 ) | ( 2 ) }
M:4/4
K:C
V:1 treble
V:2 bass octave=-1
V:1 treble
L:1/4
C2E2|]
V:2
L:1/4
C,2E,2|]"""

FOUR_NOTES = """% abcDidactyl v5
% abcD fingering 1: 12@21
% Authority: Foo Bar (1968)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% abcD fingering 2: >5135@<2222
% Authority: Bar Baz (1066)
% abcDidactyl END
X:1
T:Four Note
C:John Doe
%%score { ( 1 ) | ( 2 ) }
M:4/4
K:C
V:1 treble
V:2 bass octave=-1
V:1 treble
L:1/4
ACAe|]
V:2
L:1/4
ACAe|]
"""

A_MAJ_SCALE = """% abcDidactyl v5
% abcD fingering 1: 12312341231234543213214321321&21234123123412321432132143212&12312341231234543213214321321@54321321432132123123412312345&54321321432132123123412312345&32132143213213231231234123121
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% These are complete fingerings, with any gaps filled in.
% abcD fingering 2: >123412312312345431321432143212312312341231231432132143213212341231231234543132143214321@<432143213214321321321432132144321432132143213213214321321421432132143213214321321432132
% Authority: Hart et al. Algorithm
% abcD fingering 3: >123123412312341321421432132112123412312341232132142143213212312341231234132142143213211@<433213214321432341231234521255432132143214323412312345212532132143214321312341231234521
% Authority: Sayegh Algorithm
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

A_MAJ_SCALE_SHORT = """% abcDidactyl v5
% abcD fingering 1: 123123412312345@543213214321321
% Authority:  Beringer and Dunhill (1900)
% Transcriber: David Randolph
% Transcription date: 2016-09-13 17:24:43
% These are complete fingerings, with any gaps filled in.
% abcD fingering 2: >123412312312345@<432143213214321
% Authority: Hart et al. Algorithm
% abcD fingering 3: >123123412312341@<433213214321432
% Authority: Sayegh Algorithm
% abcDidactyl END
X:7
T:scales_a_major
C:Beringer and Dunhill
%%score { ( 1 ) | ( 2 ) }
M:4/4
K:Amaj
V:1 treble
V:2 bass octave=-1
V:1
L:1/16
A,B,CD EFGA Bcde fga2|]
V:2
L:1/16
A,B,CD EFGA Bcde fga2|]
"""

PARNCUTT_HUMAN = dict()
PARNCUTT_HUMAN['A'] = """
X:1
T:A (Op. 821 no. 1)
C:Czerny
M:C
L:1/16
Q:"Allegro"
K:C
V:1 treble
V:2 treble 
V:1
!p!egfg efde cc'bc' abga|]
V:2
[GB]4 z4 [GB]4 z4|]"""

PARNCUTT_MACHINE = dict()
PARNCUTT_MACHINE['A'] = """% abcDidactyl v5
% abcD fingering 1: ﻿24342313@xxxx
% Authority: Parncutt 1
% Transcriber: David Randolph
% Transcription date: 2018-06-19 17:24:43
% abcDidactyl END
X:1
T:A (Op. 821 no. 1)
C:Czerny
M:C
L:1/16
Q:"Allegro"
K:C
V:1 treble
V:2 treble 
V:1
!p!egfg efde cc'bc' abga|]
V:2
[GB]4 z4 [GB]4 z4|]"""
