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


# DCorpus types supported:
CORPUS_ABC = 'abc'
CORPUS_ABCD = 'abcD'
CORPUS_MUSIC_XML = 'MusicXML'
CORPUS_MIDI = 'MIDI'

HANDS_RIGHT = 1
HANDS_LEFT = 2
HANDS_EITHER = 3
HANDS_BOTH = 4

BLACK = 1
WHITE = 0

STAFF_UPPER = 0
STAFF_LOWER = 1

NATURAL_EDIT_DISTANCES = {
    ('<5', '<5'): 0,
    ('<5', '<4'): 1,
    ('<5', '<3'): 2,
    ('<5', '<2'): 3,
    ('<5', '<1'): 4,
    ('<5', '>1'): 5,
    ('<5', '>2'): 6,
    ('<5', '>3'): 7,
    ('<5', '>4'): 8,
    ('<5', '>5'): 9,

    ('<4', '<5'): 1,
    ('<4', '<4'): 0,
    ('<4', '<3'): 1,
    ('<4', '<2'): 2,
    ('<4', '<1'): 3,
    ('<4', '>1'): 4,
    ('<4', '>2'): 5,
    ('<4', '>3'): 6,
    ('<4', '>4'): 7,
    ('<4', '>5'): 8,

    ('<3', '<5'): 2,
    ('<3', '<4'): 1,
    ('<3', '<3'): 0,
    ('<3', '<2'): 1,
    ('<3', '<1'): 2,
    ('<3', '>1'): 3,
    ('<3', '>2'): 4,
    ('<3', '>3'): 5,
    ('<3', '>4'): 6,
    ('<3', '>5'): 7,

    ('<2', '<5'): 3,
    ('<2', '<4'): 2,
    ('<2', '<3'): 1,
    ('<2', '<2'): 0,
    ('<2', '<1'): 1,
    ('<2', '>1'): 2,
    ('<2', '>2'): 3,
    ('<2', '>3'): 4,
    ('<2', '>4'): 5,
    ('<2', '>5'): 6,

    ('<1', '<5'): 4,
    ('<1', '<4'): 3,
    ('<1', '<3'): 2,
    ('<1', '<2'): 1,
    ('<1', '<1'): 0,
    ('<1', '>1'): 1,
    ('<1', '>2'): 2,
    ('<1', '>3'): 3,
    ('<1', '>4'): 4,
    ('<1', '>5'): 5,

    ('>1', '<5'): 5,
    ('>1', '<4'): 4,
    ('>1', '<3'): 3,
    ('>1', '<2'): 2,
    ('>1', '<1'): 1,
    ('>1', '>1'): 0,
    ('>1', '>2'): 1,
    ('>1', '>3'): 2,
    ('>1', '>4'): 3,
    ('>1', '>5'): 4,

    ('>2', '<5'): 6,
    ('>2', '<4'): 5,
    ('>2', '<3'): 4,
    ('>2', '<2'): 3,
    ('>2', '<1'): 2,
    ('>2', '>1'): 1,
    ('>2', '>2'): 0,
    ('>2', '>3'): 1,
    ('>2', '>4'): 2,
    ('>2', '>5'): 3,

    ('>3', '<5'): 7,
    ('>3', '<4'): 6,
    ('>3', '<3'): 5,
    ('>3', '<2'): 4,
    ('>3', '<1'): 3,
    ('>3', '>1'): 2,
    ('>3', '>2'): 1,
    ('>3', '>3'): 0,
    ('>3', '>4'): 1,
    ('>3', '>5'): 2,

    ('>4', '<5'): 8,
    ('>4', '<4'): 7,
    ('>4', '<3'): 6,
    ('>4', '<2'): 5,
    ('>4', '<1'): 4,
    ('>4', '>1'): 3,
    ('>4', '>2'): 2,
    ('>4', '>3'): 1,
    ('>4', '>4'): 0,
    ('>4', '>5'): 1,

    ('>5', '<5'): 9,
    ('>5', '<4'): 8,
    ('>5', '<3'): 7,
    ('>5', '<2'): 6,
    ('>5', '<1'): 5,
    ('>5', '>1'): 4,
    ('>5', '>2'): 3,
    ('>5', '>3'): 2,
    ('>5', '>4'): 1,
    ('>5', '>5'): 0
}
