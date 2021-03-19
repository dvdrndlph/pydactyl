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
# import scamp
import mido
from music21 import pitch, note, chord
from music21.articulations import Fingering
from .DAnnotation import DAnnotation


class PigNote:
    PITCH_RE = r"^([A-G])([#b]*)(\d+)$"
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


class PigIn:
    def __init__(self, base_dir="/Users/dave/tb2/didactyl/dd/corpora/pig/PianoFingeringDataset_v1.00/"):
        self._base_dir = base_dir
        self._input_dir = base_dir + 'FingeringFiles/'
        self._midi_dir = base_dir + 'midi/'
        self._abcd_dir = base_dir + 'abcd/'
        self._fingered_note_streams = {}  # Hashed by "piece_id-annotator_id" (e.g., "009-5").

    def transform(self):
        file_re = r"^(\d+)-(\d+)_fingering.txt$"
        for root, dirs, files in os.walk(self._input_dir):
            for file in sorted(files):
                if file.endswith(".txt"):
                    mat = re.match(file_re, file)
                    if mat:
                        print(file)
                        piece_id = mat.group(1)
                        authority_id = mat.group(2)
                        file_path = os.path.join(root, file)
                        print(file_path)
                        f = open(file_path, "r")
                        mf = mido.MidiFile(type=1, ticks_per_beat=PigNote.TICKS_PER_BEAT)
                        upper = mido.MidiTrack()
                        mf.tracks.append(upper)
                        lower = mido.MidiTrack()
                        mf.tracks.append(lower)
                        # We generate the note_on and note_off messages and interleave them
                        # by relative "times" (ticks since the last event of any kind).
                        pig_tracks = [
                            {'notes_on': {}, 'notes_off': {}},
                            {'notes_on': {}, 'notes_off': {}}
                        ]
                        for line in f:
                            if line.startswith("//"):
                                continue
                            line = line.rstrip()
                            id, on, off, name, on_vel, off_vel, channel, finger = line.split()
                            knot = PigNote(id, on, off, name, on_vel, off_vel, channel, finger)
                            print(knot)
                            on_tick = knot.on_tick()
                            off_tick = knot.off_tick()
                            if on_tick not in pig_tracks[knot.channel]['notes_on']:
                                pig_tracks[knot.channel]['notes_on'][on_tick] = []
                            if off_tick not in pig_tracks[knot.channel]['notes_off']:
                                pig_tracks[knot.channel]['notes_off'][off_tick] = []
                            pig_tracks[knot.channel]['notes_on'][on_tick].append(knot)
                            pig_tracks[knot.channel]['notes_off'][off_tick].append(knot)

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

                        midi_file = piece_id + '-' + authority_id + '.mid'
                        midi_path = self._midi_dir + midi_file
                        mf.save(midi_path)
                        f.close()
                break

