__author__ = 'David Randolph'
# Copyright (c) 2021 David A. Randolph.
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
import re
import copy
import mido
from music21 import pitch
from .DAnnotation import DAnnotation
from .ABCDHeader import ABCDHeader


class PigNote:
    PITCH_RE = r"^([A-G])([#b]*)(\d+)$"
    FINGER_RE = r"(\-?)([1-5])_?(\-?)([1-5]?)"
    TICKS_PER_BEAT = 960

    def __init__(self, id, on, off, name, on_vel, off_vel, channel, finger, bpm=120):
        self.id = int(id)
        self.on = float(on)
        self.off = float(off)
        self.name = name
        self.midi_pitch = self.name_to_midi_pitch()
        self.on_vel = int(on_vel)
        self.off_vel = int(off_vel)
        self.channel = int(channel)
        self.finger = finger
        self.bpm = bpm

    def __repr__(self):
        stringy = "id:{}, on:{}, off:{}, name:{}, pitch:{}, on_vel:{}, off_vel:{}, channel:{}, finger:{}".format(
            self.id, self.on, self.off, self.name, self.midi_pitch, self.on_vel, self.off_vel, self.channel, self.finger)
        return stringy

    def tempo(self):
        tempo = mido.bpm2tempo(bpm=self.bpm)
        return tempo

    def name_to_midi_pitch(self):
        mat = re.match(PigNote.PITCH_RE, self.name)
        base_name = mat.group(1)
        accidental = mat.group(2)
        octave = mat.group(3)
        m21_accidental = ''
        for acc in accidental:
            if acc == 'b':
                m21_accidental += '-'
            else:
                m21_accidental += '#'
        m21_name = base_name + m21_accidental + octave
        pit = pitch.Pitch(m21_name)
        # print("{} ==> {} and {}".format(self.name, m21_name, pit.midi))
        return pit.midi

    def on_tick(self):
        tempo = self.tempo()
        absolute_tick = mido.second2tick(second=self.on, ticks_per_beat=PigNote.TICKS_PER_BEAT, tempo=tempo)
        return round(absolute_tick)

    def off_tick(self):
        tempo = self.tempo()
        absolute_tick = mido.second2tick(second=self.off, ticks_per_beat=PigNote.TICKS_PER_BEAT, tempo=tempo)
        return round(absolute_tick)

    def on_message(self, last_tick):
        ticks_since = self.on_tick() - last_tick
        on_msg = mido.Message('note_on', channel=self.channel, note=self.midi_pitch,
                              velocity=self.on_vel, time=ticks_since)
        return on_msg

    def off_message(self, last_tick):
        ticks_since = self.off_tick() - last_tick
        off_msg = mido.Message('note_off', channel=self.channel, note=self.midi_pitch, time=ticks_since)
        return off_msg

    def handed_abcdf(self):
        mat = re.match(PigNote.FINGER_RE, self.finger)
        if mat:
            if mat.group(1) == '-':
                strike_hand = '<'
            else:
                strike_hand = '>'
            strike_digit = mat.group(2)
            abcdf = strike_hand + strike_digit
            release_hand = '>'
            if mat.group(3) == '-':
                release_hand = '<'
            if mat.group(4):
                release_digit = mat.group(4)
                abcdf += '-' + release_hand + release_digit
            return abcdf
        else:
            raise Exception("Ill-defined PigNote fingering.")


class PigIn:
    """
    Class to manage migrating PIG piano fingering dataset to abcd header + MIDI.

    Reference:
    Nakamura, Eita, Yasuyuki Saito, and Kazuyoshi Yoshii. 2020. “Statistical Learning and
      Estimation of Piano Fingering.” Information Sciences 517: 68–85.
      https://doi.org/10.1016/j.ins.2019.12.068.

    Note that the PIG dataset does not divide the music into upper and lower staffs (or channels
    as expected in Type 1 MIDI files). Instead they define the channel by the hand used to play
    the note. This makes it impossible in some cases to define a single MIDI file (with two tracks
    or channels) to cover all PIG fingering files for the same piece. Since we require this
    sort of channel separation in Pydactyl, we would need to have more than one MIDI/abcdh
    combination for some pieces.

    To correct this, we adapt the PIG source files to make the channel assignments
    consistent. For simplicity, we simply adopt the channel assignments from the first
    fingering file for each piece. As the assignment of notes to a staff can be influenced
    by the editorial concepts of voice and to avoid unnecessary ledger lines, we often see
    quite different typesetting to express the same sequence of notes.

    Note that 17 fingering assignments in the corpus are inconsistent with the stated
    "hand part." It seems possible that the hand part does convey the staff assignment
    for the note. We changed these in the interests of simplicity. This may be ill-advised.

    For our purposes of simplifying the problem to "segregated" fingerings of melodic phrase
    segments, we think this standardization should be sufficient to provide a target rich
    corpus.

    To adopt this behavior, you must toggle it on:

         pi = PigIn(standardize=True)
         pi.transform()

    To generate multiple MIDI/abcdh files for each piece, do this:

         pi = PigIn()
         pi.transform()
    """
    def __init__(self, standardize=False, base_dir="/Users/dave/tb2/didactyl/dd/corpora/pig/PianoFingeringDataset_v1.00/"):
        self._standardize = standardize
        self._base_dir = base_dir
        self._input_dir = base_dir + 'FingeringFiles/'
        if standardize:
            self._midi_dir = base_dir + 'abcd/'
            self._abcd_dir = base_dir + 'abcd/'
        else:
            self._midi_dir = base_dir + 'individual_abcd/'
            self._abcd_dir = base_dir + 'individual_abcd/'
        self._fingered_note_streams = {}  # Hashed by "piece_id-annotator_id" (e.g., "009-5").

    @staticmethod
    def is_same_song(pt_one, pt_other):
        for track in [0, 1]:
            for msg_type in ['notes_on', 'notes_off']:
                for tick in pt_one[track][msg_type]:
                    if tick not in pt_other[track][msg_type]:
                        return False
                    if len(pt_one[track][msg_type][tick]) != len(pt_other[track][msg_type][tick]):
                        return False
        return True

    @staticmethod
    def heard_song_before(pig_tracks, pt_history):
        for annotator_id in pt_history:
            if PigIn.is_same_song(pig_tracks, pt_history[annotator_id]):
                return annotator_id
        return ""

    @staticmethod
    def sorted_pig_notes(file_path):
        f = open(file_path, "r")
        pig_notes = []
        for line in f:
            if line.startswith("//"):
                continue
            line = line.rstrip()
            msg_id, on, off, name, on_vel, off_vel, channel, finger = line.split()
            note = PigNote(msg_id, on, off, name, on_vel, off_vel, channel, finger)
            pig_notes.append(note)
        f.close()
        pig_notes.sort(key=lambda x: (x.on, x.midi_pitch))
        return pig_notes

    def transform(self):
        if not self._standardize:
            return self.transform_individual()

        file_re = r"^(\d+)-(\d+)_fingering.txt$"
        empty_pig_tracks = [
            {'notes_on': {}, 'notes_off': {}},
            {'notes_on': {}, 'notes_off': {}}
        ]
        channel_for_time_with_note = {}
        are_defining_channel = False
        annotations = {}
        pig_tracks_for_piece = {}
        for root, dirs, files in os.walk(self._input_dir):
            for file in sorted(files):
                if file.endswith(".txt"):
                    mat = re.match(file_re, file)
                    if mat:
                        piece_id = mat.group(1)
                        if self._standardize and piece_id not in channel_for_time_with_note:
                            channel_for_time_with_note[piece_id] = {}
                            are_defining_channel = True
                        if piece_id not in pig_tracks_for_piece:
                            pig_tracks_for_piece[piece_id] = {}
                        authority_id = mat.group(2)
                        file_path = os.path.join(root, file)
                        f = open(file_path, "r")
                        mf = mido.MidiFile(type=1, ticks_per_beat=PigNote.TICKS_PER_BEAT)
                        upper = mido.MidiTrack()
                        mf.tracks.append(upper)
                        lower = mido.MidiTrack()
                        mf.tracks.append(lower)
                        # We generate the note_on and note_off messages and interleave them by
                        # relative "times" (ticks since the last event of any kind on the track).
                        pig_tracks = copy.deepcopy(empty_pig_tracks)
                        upper_abcdf = ''
                        lower_abcdf = ''

                        sorted_pig_notes = PigIn.sorted_pig_notes(file_path=file_path)
                        note_count = len(sorted_pig_notes)
                        for knot in sorted_pig_notes:
                            if are_defining_channel:
                                std_channel = knot.channel
                                if knot.on not in channel_for_time_with_note[piece_id]:
                                    channel_for_time_with_note[piece_id][knot.on] = {}
                                channel_for_time_with_note[piece_id][knot.on][knot.name] = knot.channel
                            else:
                                std_channel = channel_for_time_with_note[piece_id][knot.on][knot.name]
                            knot.channel = std_channel
                            # print(knot)
                            # Absolute tick counts:
                            on_tick = knot.on_tick()
                            off_tick = knot.off_tick()
                            if on_tick not in pig_tracks[knot.channel]['notes_on']:
                                pig_tracks[knot.channel]['notes_on'][on_tick] = []
                            if off_tick not in pig_tracks[knot.channel]['notes_off']:
                                pig_tracks[knot.channel]['notes_off'][off_tick] = []
                            pig_tracks[knot.channel]['notes_on'][on_tick].append(knot)
                            pig_tracks[knot.channel]['notes_off'][off_tick].append(knot)
                            if knot.channel == 0:
                                upper_abcdf += knot.handed_abcdf()
                            else:
                                lower_abcdf += knot.handed_abcdf()
                        # Mark the whole tune as one phrase.
                        upper_abcdf += '.'
                        lower_abcdf += '.'

                        print("Note count/line count: {}".format(note_count))

                        are_defining_channel = False
                        for chan in [0, 1]:
                            ons = pig_tracks[chan]['notes_on']
                            offs = pig_tracks[chan]['notes_off']
                            ticks = list(ons.keys())
                            ticks.extend(list(offs.keys()))
                            ticks = list(set(ticks))  # Deduplicate.
                            last_tick = 0
                            for tick in sorted(ticks):
                                if tick in ons:
                                    for knot in ons[tick]:
                                        msg = knot.on_message(last_tick=last_tick)
                                        mf.tracks[chan].append(msg)
                                        last_tick = tick
                                if tick in offs:
                                    for knot in offs[tick]:
                                        msg = knot.off_message(last_tick=last_tick)
                                        mf.tracks[chan].append(msg)
                                        last_tick = tick
                        abcdf = upper_abcdf + '@' + lower_abcdf
                        authority = "PIG Annotator {}".format(authority_id)
                        annot = DAnnotation(abcdf_id=authority_id, abcdf=abcdf, authority=authority,
                                            authority_year="2019", transcriber="PIG Team")
                        last_annotator_id = PigIn.heard_song_before(pig_tracks, pig_tracks_for_piece[piece_id])
                        if last_annotator_id:
                            print("We have heard this {}-{} song before from {}" .
                                  format(piece_id, authority_id, last_annotator_id))
                            prior_midi_file = piece_id + '-' + last_annotator_id + '.mid'
                            annotations[prior_midi_file].append(annot)
                        else:
                            # Print the MIDI for the file just processed.
                            midi_file = piece_id + '-' + authority_id + '.mid'
                            midi_path = self._midi_dir + midi_file
                            mf.save(midi_path)
                            # Start spooling annotation for the new version of the piece.
                            annotations[midi_file] = []
                            annotations[midi_file].append(annot)
                            # Remember we have processed this version of the piece.
                            pig_tracks_for_piece[piece_id][authority_id] = copy.deepcopy(pig_tracks)
        for midi_file in annotations:
            abcdh = ABCDHeader(annotations=annotations[midi_file])
            abcdh_str = abcdh.__str__()
            base_name, extension = midi_file.split('.')
            abcd_file_name = base_name + '.abcd'
            abcd_path = self._abcd_dir + abcd_file_name
            abcd_fh = open(abcd_path, 'w')
            print(abcdh_str, file=abcd_fh)
            abcd_fh.close()

    def transform_individual(self):
        file_re = r"^(\d+)-(\d+)_fingering.txt$"
        empty_pig_tracks = [
            {'notes_on': {}, 'notes_off': {}},
            {'notes_on': {}, 'notes_off': {}}
        ]
        annotations = {}
        pig_tracks_for_piece = {}
        for root, dirs, files in os.walk(self._input_dir):
            for file in sorted(files):
                if file.endswith(".txt"):
                    mat = re.match(file_re, file)
                    if mat:
                        authority_id = mat.group(2)
                        piece_id = mat.group(1) + '_' + authority_id;
                        # if piece_id != '113_1':
                            # continue
                        if piece_id not in pig_tracks_for_piece:
                            pig_tracks_for_piece[piece_id] = {}
                        file_path = os.path.join(root, file)
                        mf = mido.MidiFile(type=1, ticks_per_beat=PigNote.TICKS_PER_BEAT)
                        upper = mido.MidiTrack()
                        mf.tracks.append(upper)
                        lower = mido.MidiTrack()
                        mf.tracks.append(lower)
                        # We generate the note_on and note_off messages and interleave them by
                        # relative "times" (ticks since the last event of any kind on the track).
                        pig_tracks = copy.deepcopy(empty_pig_tracks)
                        upper_abcdf = ''
                        lower_abcdf = ''
                        sorted_pig_notes = PigIn.sorted_pig_notes(file_path=file_path)
                        note_count = len(sorted_pig_notes)
                        for knot in sorted_pig_notes:
                            # print(knot)
                            on_tick = knot.on_tick()
                            off_tick = knot.off_tick()
                            if on_tick not in pig_tracks[knot.channel]['notes_on']:
                                pig_tracks[knot.channel]['notes_on'][on_tick] = []
                            if off_tick not in pig_tracks[knot.channel]['notes_off']:
                                pig_tracks[knot.channel]['notes_off'][off_tick] = []
                            pig_tracks[knot.channel]['notes_on'][on_tick].append(knot)
                            pig_tracks[knot.channel]['notes_off'][off_tick].append(knot)
                            if knot.channel == 0:
                                upper_abcdf += knot.handed_abcdf()
                            else:
                                lower_abcdf += knot.handed_abcdf()
                        # Mark the whole tune as one phrase.
                        upper_abcdf += '.'
                        lower_abcdf += '.'

                        print("Note count/line count: {}".format(note_count))

                        for chan in [0, 1]:
                            ons = pig_tracks[chan]['notes_on']
                            offs = pig_tracks[chan]['notes_off']
                            ticks = list(ons.keys())
                            ticks.extend(list(offs.keys()))
                            ticks = list(set(ticks))  # Deduplicate since the same ticks can be in on and off messages.
                            ticks = sorted(ticks)
                            last_tick = 0
                            for tick in ticks:
                                if tick in ons:
                                    for knot in ons[tick]:
                                        msg = knot.on_message(last_tick=last_tick)
                                        mf.tracks[chan].append(msg)
                                        last_tick = tick
                                if tick in offs:
                                    for knot in offs[tick]:
                                        msg = knot.off_message(last_tick=last_tick)
                                        mf.tracks[chan].append(msg)
                                        last_tick = tick
                        abcdf = upper_abcdf + '@' + lower_abcdf
                        authority = "PIG Annotator {}".format(authority_id)
                        annot = DAnnotation(abcdf_id=authority_id, abcdf=abcdf, authority=authority,
                                            authority_year="2019", transcriber="PIG Team")

                        upper_note_on_count = len(mf.tracks[0])
                        lower_note_on_count = len(mf.tracks[1])
                        print("Upper msg count: {} Lower msg_cnt: {}".format(upper_note_on_count, lower_note_on_count))
                        # Print the MIDI for the file just processed.
                        midi_file = piece_id + '.mid'
                        midi_path = self._midi_dir + midi_file
                        mf.save(midi_path)
                        # Start spooling annotation for the new version of the piece.
                        annotations[midi_file] = []
                        annotations[midi_file].append(annot)
                        # Remember we have processed this version of the piece.
                        pig_tracks_for_piece[piece_id][authority_id] = copy.deepcopy(pig_tracks)
        for midi_file in annotations:
            abcdh = ABCDHeader(annotations=annotations[midi_file])
            abcdh_str = abcdh.__str__()
            base_name, extension = midi_file.split('.')
            abcd_file_name = base_name + '.abcd'
            abcd_path = self._abcd_dir + abcd_file_name
            abcd_fh = open(abcd_path, 'w')
            print(abcdh_str, file=abcd_fh)
            abcd_fh.close()
