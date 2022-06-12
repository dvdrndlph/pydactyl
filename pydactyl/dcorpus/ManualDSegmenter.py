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


class ManualDSegmenter(DSegmenter):
    """A manual phrase-segmentation algorithm. Requires """

    LEVELS = {
        ',': 1,
        ';': 2,
        '.': 3
    }
    def __init__(self, d_annotation=None, level=None):
        super().__init__()
        self._d_annotation = d_annotation
        self._level = level
        return

    def d_annotation(self, d_annotation=None):
        if d_annotation:
            self._d_annotation = d_annotation
        return self._d_annotation

    def level(self, level=None):
        if level:
            self._level = level
        return level

    def segment_annotation(self, annotation, staff="upper"):
        hsd_segments = []
        hsd_segment = []
        note_count = self._d_annotation.score_fingering_count(staff=staff)
        hsds = annotation.handed_strike_digits(staff=staff)
        hsd_count = len(hsds)
        if hsd_count != note_count:
            raise Exception("Segmentation specifics do not match annotation to segment.")

        for i in range(hsd_count):
            hsd_segment.append(hsds[i])
            seg_mark = self._d_annotation.phrase_mark_at_index(index=i, staff=staff)
            if seg_mark and \
                    (not self._level or ManualDSegmenter.LEVELS[seg_mark] >= ManualDSegmenter.LEVELS[self._level]):
                hsd_segments.append(hsd_segment)
                hsd_segment = []
        if len(hsd_segment) > 0:
            hsd_segments.append(hsd_segment)
        return hsd_segments

    def segment_to_orderly_streams(self, d_part, offset=0):
        orderly_stream = d_part.orderly_note_stream(offset=offset)
        new_note_streams = list()
        new_note_stream = stream.Score()
        note_index = -1
        offset_adjustment = 0
        for knot in orderly_stream:
            note_index += 1
            if note_index < offset:
                continue
            # FIXME: Need to insert at appropriate offset.
            # new_note_stream.append(knot)
            # The following code needs to be validated against part with >1 segment.
            note_offset = knot.offset - offset_adjustment
            new_note_stream.insert(note_offset, knot)
            seg_mark = self._d_annotation.phrase_mark_at_index(note_index, staff=d_part.staff())
            if seg_mark and \
                    (not self._level or ManualDSegmenter.LEVELS[seg_mark] >= ManualDSegmenter.LEVELS[self._level]):
                new_note_streams.append(new_note_stream)
                new_note_stream = stream.Score()
                offset_adjustment = knot.offset
        if len(new_note_stream) > 0:
            new_note_streams.append(new_note_stream)
        return new_note_streams

    def segment_to_ordered_offset_notes(self, d_part, offset=0):
        ordered_offsets = d_part.ordered_offset_notes(offset=offset)
        segment_note_list = list()
        segment_note_lists = list()
        note_index = -1
        ql_offset_adjustment = 0
        sec_offset_adjustment = 0
        for knot in ordered_offsets:
            note_index += 1
            if note_index < offset:
                continue
            # FIXME: Need to insert at appropriate offset.
            # new_note_stream.append(knot)
            # The following code needs to be validated part with >1 segment.
            ql_offset = knot['offset'] - ql_offset_adjustment
            sec_offset = knot['second_offset'] - sec_offset_adjustment
            sec_dur = knot['second_duration']
            new_note = {'offset': ql_offset, 'second_offset': sec_offset, 'second_duration': sec_dur, 'note': knot}
            segment_note_list.append(new_note)
            seg_mark = self._d_annotation.phrase_mark_at_index(note_index, staff=d_part.staff())
            if seg_mark and \
                    (not self._level or ManualDSegmenter.LEVELS[seg_mark] >= ManualDSegmenter.LEVELS[self._level]):
                segment_note_lists.append(segment_note_list)
                ql_offset_adjustment = knot['offset']
                sec_offset_adjustment = knot['second_offset']
        if len(segment_note_list) > 0:
            segment_note_lists.append(segment_note_list)
        return segment_note_lists
