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
from music21 import *
from .DNote import DNote


class DPart:
    def __init__(self, music21_stream, segmenter=None, staff="both"):
        self._stream = music21_stream
        self._segmenter = segmenter
        self._staff = staff

    def segmenter(self):
        return self._segmenter;

    def staff(self):
        return self._staff;

    @staticmethod
    def stream_has_chords(music21_stream):
        """Returns true iff chords are present in the stream and there are
           pitches assigned to the chord. The abc importer seems to pollute
           the stream with empty chord objects.
        """
        chord_list = music21_stream.flat.getElementsByClass(chord.Chord)
        if len(chord_list) > 0:
            for cho in chord_list:
                if cho.pitchClassCardinality > 0:
                    return True
        return False

    def is_orderly(self):
        """Returns True iff this DPart contains no notes that start at the same offset
           as any other note.
        """
        if DPart.stream_has_chords(music21_stream=self._stream):
            return False

        notes = self._stream.flat.getElementsByClass(note.Note)
        for i in range(len(notes)):
            # print("{0}: {1}".format(str(notes[i].offset), str(notes[i].pitch)))
            start = notes[i].offset
            notes_at_offset = self._stream.flat.getElementsByOffset(offsetStart=start)
            if len(notes_at_offset) > 1:
                return False
        return True

    def orderly_note_stream_segments(self, offset=0):
        if self._segmenter:
            return self._segmenter.segment_to_orderly_streams(d_part=self, offset=offset)
        else:
            return [self.orderly_note_stream(offset=offset)]

    def orderly_note_stream(self, offset=0):
        """Return part as stream of notes with no notes starting at the same
           offset. Chords turned into a sequence of notes with starting points
           separated by the shortest duration (a 2048th note) ordered from
           low to high. The lowest individual note at a given offset will remain
           in place. All other notes at a given offset will be nudged to the right.
           The goal here is to provide an orderly sequence of notes that can be
           processed by Dactylers that only support monophonic streams. They can
           ignore the stacking of notes and at least take a stab at more complex
           scores (i.e., ones with chords). We also want to approximate note durations
           in case this information is useful for some models.
        """
        short_dur = duration.Duration()
        short_dur.type = '2048th'

        chords = self._stream.flat.getElementsByClass(chord.Chord)
        new_note_stream = stream.Score()
        for ch in chords:
            chord_offset = ch.offset
            note_index = 0
            for pitch_name in ch.pitchNames:
                new_note = note.Note(pitchName=pitch_name)
                new_note.offset = chord_offset + note_index * short_dur.quarterLength
                new_note_stream.append(new_note)
                note_index += 1

        notes = self._stream.flat.getElementsByClass(note.Note)
        for old_note in notes:
            new_note_stream.append(old_note)

        if not offset:
            return new_note_stream

        offset_note_stream = stream.Score()
        index = 0
        for knot in new_note_stream:
            if index < offset:
                index += 1
                continue
            offset_note_stream.append(knot)

        return offset_note_stream

    def orderly_d_notes(self, offset=0):
        m21_stream = self.orderly_note_stream(offset=offset)
        note_list = DNote.note_list(m21_stream)
        return note_list

    def length(self, offset=0):
        note_list = self.orderly_d_notes(offset=offset)
        return len(note_list)

    def pitch_range(self):
        note_stream = self.orderly_note_stream()
        low = None
        high = None
        for knot in note_stream:
            pit = knot.pitch.midi
            if not low or pit < low:
                low = pit
            if not high or pit > high:
                high = pit
        return low, high

    def is_monophonic(self):
        """Returns True iff this DPart has no notes that sound at the
           same time as other notes.
        """
        if DPart.stream_has_chords(music21_stream=self._stream):
            return False

        notes = self._stream.flat.getElementsByClass(note.Note)
        for i in range(len(notes)):
            # print("{0}: {1}".format(str(notes[i].offset), str(notes[i].pitch)))
            start = notes[i].offset
            end = start + notes[i].duration.quarterLength
            notes_in_range = self._stream.flat.getElementsByOffset(
                offsetStart=start, offsetEnd=end,
                includeEndBoundary=False,
                mustBeginInSpan=False,
                includeElementsThatEndAtStart=False,
                classList=[note.Note]
            )
            if len(notes_in_range) > 1:
                # for nir in notes_in_range:
                    # print("{0} @ {1}".format(nir, start))
                return False
        return True

    def stream(self):
        return self._stream

