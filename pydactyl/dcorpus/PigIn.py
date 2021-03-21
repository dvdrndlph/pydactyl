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

# from pprint import pprint
import os
import re
import copy
# import scamp
import mido
from music21 import pitch, note, chord
from music21.articulations import Fingering
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
        self.on_vel = int(on_vel)
        self.off_vel = int(off_vel)
        self.channel = int(channel)
        self.finger = finger
        self.bpm = bpm

    def __str__(self):
        stringy = "id:{}, on:{}, off:{}, name:{}, on_vel:{}, off_vel:{}, channel:{}, finger:{}".format(
            self.id, self.on, self.off, self.name, self.on_vel, self.off_vel, self.channel, self.finger)
        return stringy

    def tempo(self):
        tempo = mido.bpm2tempo(bpm=self.bpm)
        return tempo

    def midi_pitch(self):
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
        midi_pitch = self.midi_pitch()
        on_msg = mido.Message('note_on', channel=self.channel, note=midi_pitch,
                              velocity=self.on_vel, time=ticks_since)
        return on_msg

    def off_message(self, last_tick):
        ticks_since = self.off_tick() - last_tick
        midi_pitch = self.midi_pitch()
        off_msg = mido.Message('note_off', channel=self.channel, note=midi_pitch, time=ticks_since)
        return off_msg

    def scamp_args(self):
        midi_pitch = self.midi_pitch()
        velocity_float = self.on_vel / 127.0
        time_float = self.off - self.on
        return midi_pitch, velocity_float, time_float

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
                abcdf += release_hand + release_digit
            return abcdf
        else:
            raise Exception("Ill-defined PigNote fingering.")


class PigIn:
    def __init__(self, base_dir="/Users/dave/tb2/didactyl/dd/corpora/pig/PianoFingeringDataset_v1.00/"):
        self._base_dir = base_dir
        self._input_dir = base_dir + 'FingeringFiles/'
        self._midi_dir = base_dir + 'abcd/'
        self._abcd_dir = base_dir + 'abcd/'
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
        return False

    def transform(self):
        file_re = r"^(\d+)-(\d+)_fingering.txt$"
        empty_pig_tracks = [
            {'notes_on': {}, 'notes_off': {}},
            {'notes_on': {}, 'notes_off': {}}
        ]
        for root, dirs, files in os.walk(self._input_dir):
            prior_pig_tracks = empty_pig_tracks
            annotations = []
            is_first_file = True
            pig_tracks_for_piece = {}
            base_abcd_file_name = ''
            for file in sorted(files):
                if file.endswith(".txt"):
                    mat = re.match(file_re, file)
                    if mat:
                        piece_id = mat.group(1)
                        if piece_id not in pig_tracks_for_piece:
                            pig_tracks_for_piece[piece_id] = {}
                        authority_id = mat.group(2)
                        file_path = os.path.join(root, file)
                        if is_first_file:
                            base_abcd_file_name = piece_id + '-' + authority_id + '.abcd'
                        f = open(file_path, "r")
                        mf = mido.MidiFile(type=1, ticks_per_beat=PigNote.TICKS_PER_BEAT)
                        upper = mido.MidiTrack()
                        mf.tracks.append(upper)
                        lower = mido.MidiTrack()
                        mf.tracks.append(lower)
                        # We generate the note_on and note_off messages and interleave them
                        # by relative "times" (ticks since the last event of any kind).
                        pig_tracks = copy.deepcopy(empty_pig_tracks)
                        upper_abcdf = ''
                        lower_abcdf = ''
                        for line in f:
                            if line.startswith("//"):
                                continue
                            line = line.rstrip()
                            id, on, off, name, on_vel, off_vel, channel, finger = line.split()
                            knot = PigNote(id, on, off, name, on_vel, off_vel, channel, finger)
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
                                if tick in offs:
                                    for knot in offs[tick]:
                                        msg = knot.off_message(last_tick=last_tick)
                                        mf.tracks[chan].append(msg)
                                last_tick = tick
                        abcdf = upper_abcdf + '@' + lower_abcdf
                        authority = "PIG Annotator {}".format(authority_id)
                        annot = DAnnotation(abcdf_id=authority_id, abcdf=abcdf, authority=authority,
                                            authority_year="2019", transcriber="PIG Team")
                        abcd_file_name = piece_id + '-' + authority_id + '.abcd'
                        # First time through, we create the midi file and spool the annotation.
                        # The second time, if the song is the same, we spool the annotation.
                        # If the any song after the first song is different, we print the spooled
                        # annotations for the prior MIDI file, clear the spool, spool the latest
                        # annotation, and create the MIDI file for the new song.
                        if is_first_file:
                            is_first_file = False
                            midi_file = piece_id + '-' + authority_id + '.mid'
                            midi_path = self._midi_dir + midi_file
                            mf.save(midi_path)
                            annotations.append(annot)
                            prior_pig_tracks = copy.deepcopy(pig_tracks)
                            pig_tracks_for_piece[piece_id][authority_id] = copy.deepcopy(pig_tracks)
                            continue
                        elif PigIn.is_same_song(pig_tracks, prior_pig_tracks):
                            annotations.append(annot)
                            prior_pig_tracks = copy.deepcopy(pig_tracks)
                            continue
                        else:
                            last_annotator_id = PigIn.heard_song_before(pig_tracks, pig_tracks_for_piece[piece_id])
                            if last_annotator_id:
                                print("We have heard this {}-{} song before from {}".
                                      format(piece_id, authority_id, last_annotator_id))
                            prior_pig_tracks = copy.deepcopy(pig_tracks)
                            pig_tracks_for_piece[piece_id][authority_id] = copy.deepcopy(pig_tracks)
                            # Print the abcd file with the header corresponding to the
                            # last MIDI file generated.
                            abcdh = ABCDHeader(annotations=annotations)
                            abcdh_str = abcdh.__str__()
                            abcd_path = self._abcd_dir + base_abcd_file_name
                            abcd_fh = open(abcd_path, 'w')
                            print(abcdh_str, file=abcd_fh)
                            abcd_fh.close()

                            # Start spooling annotation for the new song.
                            annotations = []
                            annotations.append(annot)

                            # Print the MIDI for the file just processed.
                            midi_file = piece_id + '-' + authority_id + '.mid'
                            midi_path = self._midi_dir + midi_file
                            mf.save(midi_path)
                            base_abcd_file_name = abcd_file_name
                        f.close()
                # break
            abcdh = ABCDHeader(annotations=annotations)
            abcdh_str = abcdh.__str__()
            abcd_path = self._abcd_dir + base_abcd_file_name
            abcd_fh = open(abcd_path, 'w')
            print(abcdh_str, file=abcd_fh)

