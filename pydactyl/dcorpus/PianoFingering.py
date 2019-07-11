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

from music21.articulations import Fingering
from pydactyl.dcorpus.DAnnotation import DAnnotation


class PianoFingering(Fingering):
    """
    A music21 fingering for piano with full semantics per abcDF.

    >>> pf = PianoFingering()
    """
    def __init__(self, score_fingering=None, staff="upper",
                 hand=None, softened=False, dampered=False):
        self._softened = softened
        self._dampered = dampered
        default_hand = DAnnotation.default_hand(staff=staff, hand=hand)
        strike_finger = None
        if score_fingering is not None:
            self._score_fingering = score_fingering
        super().__init__(fingerNumber=strike_finger)


if __name__ == "__main__":
    import doctest
    doctest.testmod()