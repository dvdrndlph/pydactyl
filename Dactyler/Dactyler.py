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
from abc import ABC, abstractmethod


class Constant:
    # Corpus types supported:
    CORPUS_ABC = 'abc'
    CORPUS_ABCD = 'abcD'
    CORPUS_MUSIC_XML = 'MusicXML'
    CORPUS_MIDI = 'MIDI'

    HANDS_RIGHT = 1
    HANDS_LEFT = 2
    HANDS_EITHER = 3
    HANDS_BOTH = 4


class Dactyler(ABC):
    """Base class for all Didactyl algorithms."""

    # FIXME: The log should be timestamped and for the specific algorithm being used.
    LOG_FILE_PATH = '/tmp/didactyl.log'

    def __init__(self, hands=Constant.HANDS_RIGHT, chords=False):
        self.hands = hands
        self.chords = chords
        self.log = open(Dactyler.LOG_FILE_PATH, 'a')

    def squawk(self, msg):
        self.log.write(str(msg) + "\n")

    def squeak(self, msg):
        self.log.write(str(msg))

    @abstractmethod
    def advise(self, offset=0, first_finger=None):
        return

    def load_fingerings(self, path=None, query=None):
        return

    def load_corpus(self, path=None, query=None, corpus_type=Constant.CORPUS_ABC):
        return

