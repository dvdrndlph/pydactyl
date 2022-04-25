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
from music21 import note, chord, stream, duration, tempo
from .DNote import DNote


class DPart:
    def __init__(self, music21_stream, staff="both", segmenter=None):
        # All chords must have notes ordered from low to high.
        chords = music21_stream.flat.getElementsByClass(chord.Chord)
        for ch in chords:
            ch.sortAscending(inPlace=True)  # FIXME: Double accidentals will not sort by MIDI pitch.
        self._stream = music21_stream
        self._staff = staff
        self._segmenter = None
        self.segmenter(segmenter=segmenter)

    def segmenter(self, segmenter=None):
        if segmenter:
            self._segmenter = segmenter
        return self._segmenter

    def staff(self):
        return self._staff

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

    @staticmethod
    def is_pitch_in_note_list(pitch, note_list):
        for note in note_list:
            if note.pitch.midi == pitch.midi:
                return True

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
        short_dur = duration.Duration(0.0001)
        # short_dur.type = '2048th'

        chords = self._stream.flat.getElementsByClass(chord.Chord)
        chord_cnt = len(chords)
        chord_stream = chords.stream()
        notes_at_offset = {}
        chord_notes = []
        offsetted_notes = []
        for ch in chords:
            chord_offset = ch.getOffsetBySite(chord_stream)
            if chord_offset not in notes_at_offset:
                notes_at_offset[chord_offset] = []
            for pit in ch.pitches:
                new_note = note.Note(pit)
                new_note.quarterLength = ch.quarterLength
                if not DPart.is_pitch_in_note_list(pit, notes_at_offset[chord_offset]):
                    notes_at_offset[chord_offset].append(new_note)
                    offsetted_notes.append(new_note)
                    chord_notes.append(new_note)
                else:
                    print("music21 MIDI spawned extra note in chord.")

        offset_cnt = len(notes_at_offset)
        if chord_cnt != offset_cnt:
            print("{} chord(s) subsumed by other(s).".format(chord_cnt - offset_cnt))

        offsetted_cnt = len(offsetted_notes)
        nao_cnt = 0
        for note_offset in notes_at_offset:
            nao_cnt += len(notes_at_offset[note_offset])
        if offsetted_cnt != nao_cnt:
            raise Exception("Count mismatch for chord notes at offsets.")

        notes = self._stream.flat.getElementsByClass(note.Note)
        note_list = list(notes)
        note_stream = notes.stream()
        for old_note in note_list:
            note_offset = old_note.getOffsetBySite(note_stream)
            if note_offset not in notes_at_offset:
                notes_at_offset[note_offset] = [old_note]
            else:
                notes_at_offset[note_offset].append(old_note)
            offsetted_notes.append(old_note)

        stream_index = 0
        prior_stream_size = 0
        new_note_stream = stream.Score()
        epoch_num = 1
        for note_offset in sorted(notes_at_offset):
            offset_notes = notes_at_offset[note_offset]
            notes_len = len(offset_notes)
            if len(offset_notes) > 1:
                sorted_notes = sorted(offset_notes, key=lambda x: x.pitch.midi)
            else:
                sorted_notes = offset_notes
            sorted_notes_len = len(sorted_notes)
            if sorted_notes_len != notes_len:
                raise Exception("Sorting has gone off the rails.")
            note_index = 0
            prior_old_note = None
            for old_note in sorted_notes:
                if prior_old_note and old_note.pitch.midi == prior_old_note.pitch.midi:
                    raise Exception("Duplicate {} pitches at epoch {} (offset {}) in part near note index {}.".format(
                        prior_old_note.pitch, epoch_num, note_offset, stream_index))
                new_note_offset = note_offset + note_index * short_dur.quarterLength
                new_note_stream.insert(new_note_offset, old_note)
                new_stream_size = len(new_note_stream.elements)
                if new_stream_size != prior_stream_size + 1:
                    raise Exception("Bad stream insert at index {} of note{}".format(stream_index, old_note))
                prior_stream_size = new_stream_size
                prior_old_note = old_note
                # print("Note offset: {}".format(note_offset))
                note_index += 1
                stream_index += 1
            epoch_num += 1

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

    def orderly_d_note_segments(self, offset=0):
        orderly_streams = self._segmenter.segment_to_orderly_streams(d_part=self, offset=offset)
        d_note_lists = []
        for ostrm in orderly_streams:
            note_list = DNote.note_list(ostrm)
            d_note_lists.append(note_list)
        return d_note_lists

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
                #     print("{0} @ {1}".format(nir, start))
                return False
        return True

    def stream(self):
        return self._stream
