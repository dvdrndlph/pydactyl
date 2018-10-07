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
from music21 import *
from .DSegmenter import DSegmenter


class OneDSegmenter(DSegmenter):
    """A degenerate phrase segmenter that always returns one segment."""

    def __init__(self, d_annotation=None):
        self._d_annotation = d_annotation
        return

    def segment_to_orderly_streams(self, d_part, offset=0):
        orderly_stream = d_part.orderly_note_stream(offset=offset)
        new_note_stream = stream.Score()
        note_index = -1
        for knot in orderly_stream:
            note_index += 1
            if note_index < offset:
                continue
            new_note_stream.append(knot)

        return [new_note_stream]
