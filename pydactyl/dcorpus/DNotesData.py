__author__ = 'David Randolph'
# Copyright (c) 2023 David A. Randolph.
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
from music21 import stream, note, chord, midi


class DNotesData:
    def __init__(self, notes, staff, d_score=None):
        self.notes = notes
        self.staff = staff
        self.d_score = d_score
        self.chordified = None
        self.chordified_offsets = None
        self.chordified_list = None
        self.consonant_map = list()
        # self.chordify() # PIG 108 cannot be processed
        self.pits_sounding_at = self.my_chordify()
        # self.map_strike_consonants()

    def my_chordify(self):
        pits_on_at = dict()
        pits_currently_on = set()
        pits_off_at = dict()
        note_index = 0
        for note_data in self.notes:
            pit = note_data['note'].pitch
            onset = note_data['second_offset']
            offset = onset + note_data['second_duration']

            offsets_purged = list()
            for known_offset in pits_off_at:
                if onset >= known_offset:
                    for dead_pit in pits_off_at[known_offset]:
                        try:
                            pits_currently_on.remove(dead_pit)
                        except Exception:
                            print("Missing {} pitch to turn off at {} second mark, detected at note index {}.".format(
                                dead_pit, known_offset, note_index))

                    offsets_purged.append(known_offset)
            for purged_offset in offsets_purged:
                pits_off_at.pop(purged_offset)

            if onset not in pits_on_at:
                pits_on_at[onset] = set()
            if offset not in pits_off_at:
                pits_off_at[offset] = set()
            pits_currently_on.add(pit)
            pits_off_at[offset].add(pit)
            pits_on_at[onset].update(pits_currently_on)

            note_index += 1

        return pits_on_at

    def chordify(self):
        note_stream = stream.Stream()
        for knot in self.notes:
            note_stream.insert(knot['offset'], knot['note'])
        self.chordified = note_stream.chordify()
        # self.chordified.show('text')

    def map_strike_consonants(self):
        """
        Separately store higher pitches and lower pitches sounding at the same time
        each note is struck. The idea is capture constraints from all keys
        still held down, not just those that are simultaneously struck.

        FIXME: Chordify gives us enough information to quantize by the chord threshold
        (the grace period afforded to conclude two notes are struck simultaneously).
        As implemented here, we will get very different results from symbolic and
        performance input data. The chording features should be more or less consistent.
        The key thing we are trying to capture here is a constraint placed on passing
        motion. This will be screamingly apparent in symbolic data: each individual
        note will reflect the same state of affairs. But in performance data, the
        earliest note will appear unencumbered, the second constrained by one note,
        the third by two, etc. We could scrutinize the ties in chordify and create
        a more unified view. This also seems to call for a "training type" feature,
        as the semantic differences between the two types of data are salient.

        NOTE: PIG score #108 is clearly not derived from a human performance. So bigger
        FIXME: Fix apparent bug in music21 chordify() method. For now, we implement
        our own chordify-like thing.
        """
        chord_list = list()
        offsets = list()
        for ch in self.chordified.getElementsByClass(chord.Chord):
            chord_list.append(ch)
            offsets.append(ch.offset)
        self.chordified_list = chord_list
        self.chordified_offsets = offsets
        chord_i = 0
        current_chord = chord_list[chord_i]
        for note_i in range(len(self.notes)):
            knot = self.notes[note_i]['note']
            knot_pit = knot.pitch
            while knot_pit not in current_chord.pitches:
                chord_i += 1
                if chord_i >= len(chord_list):
                    raise Exception("chordify() seems to have dropped a note.")
                current_chord = chord_list[chord_i]
            higher_pits = list()
            lower_pits = list()
            for pit in current_chord.pitches:
                if pit.midi > knot_pit.midi:
                    higher_pits.append(pit)
                elif pit.midi < knot_pit.midi:
                    lower_pits.append(pit)
            details = {
                'pitch': knot_pit,
                'higher_pitches': higher_pits,
                'lower_pitches': lower_pits
            }
            self.consonant_map.append(details)

    def higher_consonant_note_count(self, i):
        note_count = len(self.consonant_map[i]['higher_pitches'])
        return note_count

    def lower_consonant_note_count(self, i):
        note_count = len(self.consonant_map[i]['lower_pitches'])
        return note_count

    def consonant_count_feature_str(self, i):
        higher_count = self.higher_consonant_note_count(i)
        lower_count = self.lower_consonant_note_count(i)
        feature_str = "{}_{}".format(lower_count, higher_count)
        return feature_str


