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

import os
DACTYLER_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = DACTYLER_DIR + '/../../data'

# dcorpus types supported:
CORPUS_ABC = 'abc'
CORPUS_ABCD = 'abcD'
CORPUS_MUSIC_XML = 'MusicXML'
CORPUS_MIDI = 'MIDI'
MIDI_FILE_RE = r"Standard MIDI data.*format 1.*using 2 tracks"
# MIDI_MIME_RE = r"^audio/midi$"


THUMB = 1
INDEX = 2
MIDDLE = 3
RING = 4
LITTLE = 5

HANDS_RIGHT = 1
HANDS_LEFT = 2
HANDS_EITHER = 3
HANDS_BOTH = 4

BLACK = 1
WHITE = 0

STAFF_UPPER = 0
STAFF_LOWER = 1

PIVOT_EDIT_DISTANCES = {
    ('x', '<5'): 0,
    ('x', '<4'): 0,
    ('x', '<3'): 0,
    ('x', '<2'): 0,
    ('x', '<1'): 0,
    ('x', '>1'): 0,
    ('x', '>2'): 0,
    ('x', '>3'): 0,
    ('x', '>4'): 0,
    ('x', '>5'): 0,
    ('<1', 'x'): 0,
    ('<2', 'x'): 0,
    ('<3', 'x'): 0,
    ('<4', 'x'): 0,
    ('<5', 'x'): 0,
    ('>1', 'x'): 0,
    ('>2', 'x'): 0,
    ('>3', 'x'): 0,
    ('>4', 'x'): 0,
    ('>5', 'x'): 0,
    ('x', 'x'): 0,
    ('<5', '<5'): 0,
    ('<5', '<4'): 1,
    ('<5', '<3'): 2,
    ('<5', '<2'): 3,
    ('<5', '<1'): 7,
    ('<5', '>1'): 14,
    ('<5', '>2'): 15,
    ('<5', '>3'): 16,
    ('<5', '>4'): 17,
    ('<5', '>5'): 18,

    ('<4', '<5'): 1,
    ('<4', '<4'): 0,
    ('<4', '<3'): 1,
    ('<4', '<2'): 2,
    ('<4', '<1'): 6,
    ('<4', '>1'): 13,
    ('<4', '>2'): 14,
    ('<4', '>3'): 15,
    ('<4', '>4'): 16,
    ('<4', '>5'): 17,

    ('<3', '<5'): 2,
    ('<3', '<4'): 1,
    ('<3', '<3'): 0,
    ('<3', '<2'): 1,
    ('<3', '<1'): 5,
    ('<3', '>1'): 12,
    ('<3', '>2'): 13,
    ('<3', '>3'): 14,
    ('<3', '>4'): 15,
    ('<3', '>5'): 16,

    ('<2', '<5'): 3,
    ('<2', '<4'): 2,
    ('<2', '<3'): 1,
    ('<2', '<2'): 0,
    ('<2', '<1'): 4,
    ('<2', '>1'): 11,
    ('<2', '>2'): 12,
    ('<2', '>3'): 13,
    ('<2', '>4'): 14,
    ('<2', '>5'): 15,

    ('<1', '<5'): 7,
    ('<1', '<4'): 6,
    ('<1', '<3'): 5,
    ('<1', '<2'): 4,
    ('<1', '<1'): 0,
    ('<1', '>1'): 10,
    ('<1', '>2'): 11,
    ('<1', '>3'): 12,
    ('<1', '>4'): 13,
    ('<1', '>5'): 14,

    ('>1', '<5'): 14,
    ('>1', '<4'): 13,
    ('>1', '<3'): 12,
    ('>1', '<2'): 11,
    ('>1', '<1'): 10,
    ('>1', '>1'): 0,
    ('>1', '>2'): 4,
    ('>1', '>3'): 5,
    ('>1', '>4'): 6,
    ('>1', '>5'): 7,

    ('>2', '<5'): 15,
    ('>2', '<4'): 14,
    ('>2', '<3'): 13,
    ('>2', '<2'): 12,
    ('>2', '<1'): 11,
    ('>2', '>1'): 4,
    ('>2', '>2'): 0,
    ('>2', '>3'): 1,
    ('>2', '>4'): 2,
    ('>2', '>5'): 3,

    ('>3', '<5'): 16,
    ('>3', '<4'): 15,
    ('>3', '<3'): 14,
    ('>3', '<2'): 13,
    ('>3', '<1'): 12,
    ('>3', '>1'): 5,
    ('>3', '>2'): 1,
    ('>3', '>3'): 0,
    ('>3', '>4'): 1,
    ('>3', '>5'): 2,

    ('>4', '<5'): 17,
    ('>4', '<4'): 16,
    ('>4', '<3'): 15,
    ('>4', '<2'): 14,
    ('>4', '<1'): 13,
    ('>4', '>1'): 6,
    ('>4', '>2'): 2,
    ('>4', '>3'): 1,
    ('>4', '>4'): 0,
    ('>4', '>5'): 1,

    ('>5', '<5'): 18,
    ('>5', '<4'): 17,
    ('>5', '<3'): 16,
    ('>5', '<2'): 15,
    ('>5', '<1'): 14,
    ('>5', '>1'): 7,
    ('>5', '>2'): 3,
    ('>5', '>3'): 2,
    ('>5', '>4'): 1,
    ('>5', '>5'): 0
}

MAX_NATURAL_EDIT_DISTANCE = 9
NATURAL_EDIT_DISTANCES = {
    ('x', '<5'): 0,
    ('x', '<4'): 0,
    ('x', '<3'): 0,
    ('x', '<2'): 0,
    ('x', '<1'): 0,
    ('x', '>1'): 0,
    ('x', '>2'): 0,
    ('x', '>3'): 0,
    ('x', '>4'): 0,
    ('x', '>5'): 0,
    ('<1', 'x'): 0,
    ('<2', 'x'): 0,
    ('<3', 'x'): 0,
    ('<4', 'x'): 0,
    ('<5', 'x'): 0,
    ('>1', 'x'): 0,
    ('>2', 'x'): 0,
    ('>3', 'x'): 0,
    ('>4', 'x'): 0,
    ('>5', 'x'): 0,
    ('x', 'x'): 0,

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
