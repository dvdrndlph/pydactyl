__author__ = 'David Randolph'
# Copyright (c) 2021, 2022 David A. Randolph.
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
import subprocess
import shutil
import re
import copy
import mido
from scipy.stats import ttest_rel
from sklearn_crfsuite import metrics
from music21 import pitch, note
from pathlib import Path
from .DAnnotation import DAnnotation
from .ABCDHeader import ABCDHeader
from .DScore import DScore

MICROSECONDS_PER_BEAT = 500000  # at 120 bpm
MS_PER_BEAT = MICROSECONDS_PER_BEAT / 1000
REPO_DIR = '/Users/dave/tb2/'
PIG_BASE_DIR = REPO_DIR + 'didactyl/dd/corpora/pig/'
PIG_RESULT_FHMM3_DIR = PIG_BASE_DIR + 'Result_FHMM3/'
PIG_DATASET_DIR = PIG_BASE_DIR + 'PianoFingeringDataset_v1.00/'
PIG_FINGERING_DIR = PIG_DATASET_DIR + 'FingeringFiles/'
PIG_ABCD_DIR = PIG_DATASET_DIR + 'individual_abcd/'
PIG_MERGED_ABCD_DIR = PIG_DATASET_DIR + 'abcd/'
PIG_STD_DIR = PIG_DATASET_DIR + 'std_pig/'
PIG_SEGREGATED_DATASET_DIR = PIG_DATASET_DIR + 'segregated_pig/'
PIG_SEGREGATED_FINGERING_DIR = PIG_SEGREGATED_DATASET_DIR + 'FingeringFiles/'
PIG_SEGREGATED_ABCD_DIR = PIG_SEGREGATED_DATASET_DIR + 'individual_abcd/'
PIG_SEGREGATED_STD_DIR = PIG_SEGREGATED_DATASET_DIR + 'std_pig/'
PIG_BIN_DIR = PIG_BASE_DIR + 'SourceCode/Binary/'
SIMPLE_MATCH_RATE_CMD = PIG_BIN_DIR + 'Evaluate_SimpleMatchRate'
COMPLEX_MATCH_RATES_CMD = PIG_BIN_DIR + 'Evaluate_MultipleGroundTruth'
PIG_SCRIPT_DIR = PIG_BASE_DIR + 'SourceCode/'
PIG_PREDICTION_DIR = '/tmp/pig_test/'
PIG_FILE_SUFFIX = '_fingering.txt'
PIG_FILE_RE = r"^((\d+)-(\d+))_fingering.txt$"
PIG_FILE_NAME_LOC = 1
PIG_FILE_PIECE_LOC = 2
PIG_FILE_ANNOT_LOC = 3
PIG_STRIKE_LABELS = ['-1', '1', '-2', '2', '-3', '3', '-4', '4', '-5', '5']

NAKAMURA_MODEL_CMDS = {
    'fhmm1': PIG_SCRIPT_DIR + 'run_FHMM1.sh',
    'fhmm2': PIG_SCRIPT_DIR + 'run_FHMM2.sh',
    'fhmm3': PIG_SCRIPT_DIR + 'run_FHMM3.sh',
    'chmm':  PIG_SCRIPT_DIR + 'run_CHMM.sh'
}
NAKAMURA_METRICS = ('general', 'highest', 'soft', 'recomb')
NAKAMURA_METRIC_SUBSCRIPTS = {
    'general': 'gen',
    'highest': 'high',
    'soft': 'soft',
    'recomb': 'rec'
}


class PigNote:
    PITCH_RE = r"^([A-G])([#b]*)(\d+)$"
    M21_PITCH_RE = r"^([A-G])([#-]*)(\d+)$"
    FINGER_RE = r"(\-?)([1-5])_?(\-?)([1-5]?)"
    HANDED_DIGIT_RE = r'([><]+)([1-5]+)'
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

    def to_file_line(self):
        stringy = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            self.id, self.on, self.off, self.name, self.on_vel, self.off_vel, self.channel, self.finger)
        return stringy

    def strike_pig_finger(self):
        mat = re.match(PigNote.FINGER_RE, self.finger)
        if mat:
            strike_piggy = "{}{}".format(mat.group(1), mat.group(2))
            return strike_piggy
        raise Exception("Ill-formed PIG finger: {}".format(self.finger))

    @staticmethod
    def header_line():
        return "//Version: PianoFingering_v170101\n"

    def tempo(self):
        tempo = mido.bpm2tempo(bpm=self.bpm)
        return tempo

    @staticmethod
    def abcdf_to_pig_fingering(handed_digit):
        mat = re.match(PigNote.HANDED_DIGIT_RE, handed_digit)
        if mat:
            if mat.group(1) == '<':
                hand = '-'
            else:
                hand = ''
            digit = mat.group(2)
            pig_fingering = hand + digit
            return pig_fingering
        raise Exception("Handed digit {} is ill-formed".format(handed_digit))

    @staticmethod
    def m21_name_to_pig_name(m21_note_name):
        mat = re.match(PigNote.M21_PITCH_RE, m21_note_name)
        if mat:
            base_name = mat.group(1)
            accidental = mat.group(2)
            octave = mat.group(3)
            pig_accidental = ''
            for acc in accidental:
                if acc == '-':
                    pig_accidental += 'b'
                else:
                    pig_accidental += '#'
            pig_name = base_name + pig_accidental + octave
            return pig_name
        raise Exception("Bad music21 pitch name: {}".format(m21_note_name))

    @staticmethod
    def pig_name_to_m21_name(pig_note_name):
        mat = re.match(PigNote.PITCH_RE, pig_note_name)
        if mat:
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
            return m21_name
        else:
            raise Exception("Bad PigNote name: {}.".format(pig_note_name))

    def name_to_midi_pitch(self):
        m21_name = PigNote.pig_name_to_m21_name(self.name)
        pit = pitch.Pitch(m21_name)
        # print("{} ==> {} and {}".format(self.name, m21_name, pit.midi))
        return pit.midi

    @staticmethod
    def midi_pitch_to_name(midi_pitch):
        p = pitch.Pitch(midi=midi_pitch)
        m21_note_name = p.name + str(p.octave)
        pig_note_name = PigNote.m21_name_to_pig_name(m21_note_name=m21_note_name)
        return pig_note_name

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
    def __init__(self, standardize=False, base_dir=PIG_DATASET_DIR, start_over=False):
        self._standardize = standardize
        self._base_dir = base_dir
        self._input_dir = base_dir + 'FingeringFiles/'
        if standardize:
            self._midi_dir = base_dir + 'abcd/'
            self._abcd_dir = base_dir + 'abcd/'
            self._std_pig_dir = None
        else:
            self._midi_dir = base_dir + 'individual_abcd/'
            self._abcd_dir = base_dir + 'individual_abcd/'
            self._std_pig_dir = base_dir + 'std_pig/'
            PigIn.mkdir_if_missing(self._std_pig_dir, make_missing=start_over)
        self._fingered_note_streams = {}  # Hashed by "piece_id-annotator_id" (e.g., "009-5").
        PigIn.mkdir_if_missing(self._midi_dir, make_missing=start_over)
        PigIn.mkdir_if_missing(self._abcd_dir, make_missing=start_over)

    @staticmethod
    def mkdir_if_missing(path_str, make_missing=False):
        path = Path(path_str)
        if make_missing and path.is_dir():
            shutil.rmtree(path_str)
        if not path.is_dir():
            os.makedirs(path_str)

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
        pig_notes = PigIn.pig_notes(file_path)
        pig_notes.sort(key=lambda x: (x.on, x.midi_pitch))
        return pig_notes

    @staticmethod
    def pig_notes(file_path):
        f = open(file_path, "r")
        pig_notes = []
        for line in f:
            if line.startswith("//"):
                continue
            line = line.rstrip()
            msg_id, on, off, name, on_vel, off_vel, channel, finger = line.split()
            pig_note = PigNote(msg_id, on, off, name, on_vel, off_vel, channel, finger)
            pig_notes.append(pig_note)
        f.close()
        return pig_notes

    def transform(self):
        if not self._standardize:
            return self.transform_individual()

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
                if file.endswith(PIG_FILE_SUFFIX):
                    mat = re.match(PIG_FILE_RE, file)
                    if mat:
                        piece_id = mat.group(PIG_FILE_PIECE_LOC)
                        if self._standardize and piece_id not in channel_for_time_with_note:
                            channel_for_time_with_note[piece_id] = {}
                            are_defining_channel = True
                        if piece_id not in pig_tracks_for_piece:
                            pig_tracks_for_piece[piece_id] = {}
                        authority_id = mat.group(PIG_FILE_ANNOT_LOC)
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
        empty_pig_tracks = [
            {'notes_on': {}, 'notes_off': {}},
            {'notes_on': {}, 'notes_off': {}}
        ]
        annotations = {}
        pig_tracks_for_file = {}
        for root, dirs, files in os.walk(self._input_dir):
            for file in sorted(files):
                if file.endswith(PIG_FILE_SUFFIX):
                    mat = re.match(PIG_FILE_RE, file)
                    if mat:
                        authority_id = mat.group(PIG_FILE_ANNOT_LOC)
                        file_id = mat.group(PIG_FILE_NAME_LOC)
                        # if piece_id != '113_1':
                            # continue
                        if file_id not in pig_tracks_for_file:
                            pig_tracks_for_file[file_id] = {}
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
                        midi_file = file_id + '.mid'
                        midi_path = self._midi_dir + midi_file
                        mf.save(midi_path)
                        annotations[file_id] = []
                        annotations[file_id].append(annot)
                        # Remember we have processed this version of the piece.
                        pig_tracks_for_file[file_id][authority_id] = copy.deepcopy(pig_tracks)
        for file_id in annotations:
            abcdh = ABCDHeader(annotations=annotations[file_id])
            abcdh_str = abcdh.__str__()
            midi_file = file_id + '.mid'
            midi_path = self._midi_dir + midi_file
            abcd_file = file_id + '.abcd'
            abcd_path = self._abcd_dir + abcd_file
            abcd_fh = open(abcd_path, 'w')
            print(abcdh_str, file=abcd_fh)
            abcd_fh.close()
            m21_score = DScore.score_via_midi(corpus_path=midi_path)
            d_score = DScore(music21_stream=m21_score, abcd_header=abcdh, title=file_id)
            pout = PigOut(d_score=d_score)
            std_pig_file = file_id + PIG_FILE_SUFFIX
            std_pig_path = self._std_pig_dir + std_pig_file
            pout.transform(to_file=std_pig_path)


class PigOut:
    """
    Class to export a DScore to "standardized" PIG FingerFile format. The _fingering.txt files
    provided in the PIG dataset are sorted by note start times (in seconds offset from the
    beginning of the piece). But in the event of ties, it does not fix the notes in any
    particular order. This conflicts with the way ABCD works and aligns fingering annotations
    with notes. We therefore enforce a "left-to-right, low-to-high" total order in the output
    produced here.

    Note that transforming an original PIG file to a DScore via the PigIn class above and
    subsequently transforming that DScore to a PIG file using this class, will NOT produce
    output identical to the original. The notes will be reordered and renumbered. Also,
    offset times, while similar, will not be exactly the same. There are three known potential
    sources for these discrepancies. First, the original PIG files are translated to MIDI format,
    via midi.translate.midiFileToStream(mf=mf, quantizePost=False), provided by music21.
    Time resolution may deviate by the MICROSECONDS_PER_BEAT setting for the MIDI file.
    It may not be possible to capture the full resolution dictated in the PIG source file.
    A more significant problem is with quantization performed by music21, as it defines a notion
    of "quarterLength" in its efforts to create a symbolic representation of the music.
    Durations are therefore modified as needed to support no more than 1/2048th note
    granularity. This can can cause offset timestamps to shift. Finally, for historical
    reasons, within each channel (staff), our code imposes an explicit total order in which
    no two notes are allowed to start at the same time. We accomplish this by simply shaving
    1/2048th-note durations from the beginning of notes in a chord to ensure a total order.
    (Performing this operation was done to allow models that do not take chords into effect to
    at least be able to process music that contains chords. FIXME: This was a kludge, and we
    are nearing the point when we will be modifying these models to deal with this issue
    more intelligently.)
    """
    def __init__(self, d_score: DScore):
        self._d_score = d_score

    @staticmethod
    def simple_match_rate(gt_pig_path, pred_pig_path):
        result = subprocess.run([SIMPLE_MATCH_RATE_CMD, gt_pig_path, pred_pig_path],
                                capture_output=True, encoding='utf-8')
        mat = re.match(r'MatchRate:\s*(\d+)/(\d+)', result.stdout)
        match_rate = {}
        if mat:
            match_rate['match_count'] = int(mat.group(1))
            match_rate['note_count'] = int(mat.group(2))
            match_rate['rate'] = 1.0*match_rate['match_count'] / match_rate['note_count']
        # print(result.stdout)
        return match_rate

    @staticmethod
    def simple_match_rates(gt_pig_paths, pred_pig_paths):
        match_rates = list()
        for i in range(pred_pig_paths):
            match_rate = PigOut.simple_match_rate(gt_pig_path=gt_pig_paths[i], pred_pig_path=pred_pig_paths[i])
            match_rates.append(match_rate['rate'])
        return match_rates

    @staticmethod
    def simple_match_rates_paired_t_test(self, gt_pig_paths=None, better_pig_paths=None, worse_pig_paths=None,
                                         gt_pig_dir=None, better_pig_dir=None, worse_pig_dir=None):
        if gt_pig_dir:
            gt_pig_paths = PigOut.pig_paths_in_dir(gt_pig_dir)
        if better_pig_dir:
            better_pig_paths = PigOut.pig_paths_in_dir(better_pig_dir)
        if worse_pig_dir:
            worse_pig_paths = PigOut.pig_paths_in_dir(worse_pig_dir)
        better_rates = PigOut.simple_match_rate(gt_pig_paths=gt_pig_paths, pred_pig_path=better_pig_paths)
        worse_rates = PigOut.simple_match_rate(gt_pig_paths=gt_pig_paths, pred_pig_path=worse_pig_paths)
        stat, p_val = ttest_rel(better_rates, worse_rates)
        return stat, p_val

    @staticmethod
    def pig_paths_in_dir(pig_dir):
        paths = list()
        for file_name in sorted(os.listdir(pig_dir)):
            if file_name.endswith(PIG_FILE_SUFFIX):
                path = "{}/{}".format(pig_dir, file_name)
                paths.append(path)
        return paths

    @staticmethod
    def my_single_prediction_m_gen(gt_pig_paths, pred_pig_path):
        pred_pig_notes = PigIn.pig_notes(file_path=pred_pig_path)
        note_count = len(pred_pig_notes)
        gt_note_total_count = note_count*len(gt_pig_paths)
        gt_match_counts = list()
        match_total_count = 0
        for gt_pp in gt_pig_paths:
            gt_match_count = 0
            gt_pig_notes = PigIn.pig_notes(file_path=gt_pp)
            if len(gt_pig_notes) != note_count:
                raise Exception("Files are of unequal length.")
            for i in range(note_count):
                pred_note = pred_pig_notes[i]
                pred_strike_finger = pred_note.strike_pig_finger()
                gt_note = gt_pig_notes[i]
                gt_strike_finger = gt_note.strike_pig_finger()
                if gt_strike_finger == pred_strike_finger:
                    gt_match_count += 1
            gt_match_counts.append(gt_match_count)
            match_total_count += gt_match_count
        m_gen = match_total_count/gt_note_total_count
        return match_total_count, gt_note_total_count, m_gen

    @staticmethod
    def my_single_prediction_m(gt_pig_paths, pred_pig_path):
        pred_pig_notes = PigIn.pig_notes(file_path=pred_pig_path)
        note_count = len(pred_pig_notes)
        gt_annot_total_count = note_count*len(gt_pig_paths)
        gt_match_counts = list()
        match_total_counts = {
            'general': 0,
            'highest': 0,
            'soft': 0,
            'recomb': 0
        }
        m = {
            'general': 0,
            'highest': 0,
            'soft': 0,
            'recomb': 0
        }
        highest_gt_match_count = 0
        match_bits = [0]*note_count
        for gt_pp in gt_pig_paths:
            gt_match_count = 0
            gt_pig_notes = PigIn.pig_notes(file_path=gt_pp)
            if len(gt_pig_notes) != note_count:
                raise Exception("Files are of unequal length.")
            for i in range(note_count):
                pred_note = pred_pig_notes[i]
                pred_strike_finger = pred_note.strike_pig_finger()
                gt_note = gt_pig_notes[i]
                gt_strike_finger = gt_note.strike_pig_finger()
                if gt_strike_finger == pred_strike_finger:
                    gt_match_count += 1
                    match_bits[i] = 1
            if gt_match_count > highest_gt_match_count:
                highest_gt_match_count = gt_match_count
            gt_match_counts.append(gt_match_count)
            match_total_counts['general'] += gt_match_count
        m['general'] = match_total_counts['general'] / gt_annot_total_count
        m['highest'] = highest_gt_match_count / note_count
        m['soft'] = match_bits.count(1) / note_count
        return m

    @staticmethod
    def single_prediction_complex_match_rates(gt_pig_paths, pred_pig_path):
        cmd_tokens = list()
        cmd_tokens.append(COMPLEX_MATCH_RATES_CMD)
        cmd_tokens.append(str(len(gt_pig_paths)))
        for gt_pig_path in gt_pig_paths:
            cmd_tokens.append(gt_pig_path)
        cmd_tokens.append(pred_pig_path)
        result = subprocess.run(cmd_tokens, capture_output=True, encoding='utf-8')
        mat = re.match(r'General,Highest,Soft,Recomb:\s*([\.\d]+)\s+([\.\d]+)\s+([\.\d]+)\s+([\.\d]+)', result.stdout)
        result = {}
        if mat:
            result['general'] = float(mat.group(1))
            result['highest'] = float(mat.group(2))
            result['soft'] = float(mat.group(3))
            result['recomb'] = float(mat.group(4))
        else:
            # raise Exception("No matchy.")
            return None
        # print(result.stdout)
        return result

    @staticmethod
    def run_hmm(in_fin, out_fin, model='fhmm3'):
        model = model.lower()
        cmd = NAKAMURA_MODEL_CMDS[model]

        out_path = Path(out_fin)
        if out_path.is_file():
            os.remove(out_fin)

        result = subprocess.run([cmd, in_fin, out_fin], capture_output=True, encoding='utf-8')
        if result.returncode != 0:
            raise Exception("{} returned non-zero exit code: {}.".format(cmd, result.stderr))
        if not out_path.is_file():
            raise Exception("{} created no output file.".format(cmd))
        stats = os.stat(out_fin)
        if stats.st_size < 100:
            raise Exception("{} generated too little data in {} file for {} file.".format(cmd, out_fin, in_fin))

    @staticmethod
    def get_max_model_str_len():
        model_str = 'std_pig FHMM3 (unweighted)'
        max_model_str_len = len(model_str)
        return max_model_str_len

    @staticmethod
    def output_nakamura_metrics_heading(output_type="text", decimals=4):
        if output_type == 'text':
            max_model_str_len = PigOut.get_max_model_str_len()
            output_str = "{:>" + str(max_model_str_len) + '}'
            output_str = output_str.format("Method") + "\t"
            width = decimals + 2
            format_str = '{:>' + str(width) + '}'

            for metric in NAKAMURA_METRICS:
                output_str += format_str.format(NAKAMURA_METRIC_SUBSCRIPTS[metric]) + "\t"
            print(output_str)

    @staticmethod
    def output_nakamura_metrics(results, corpus_name, model, weight, output_type="text", decimals=4):
        output_str = ''
        metrics = NAKAMURA_METRICS
        rounded_results = dict()
        for metric in metrics:
            rounded_results[metric] = round(results[metric], ndigits=decimals)
        if output_type == 'text':
            weight_str = '(unweighted)'
            if weight:
                weight_str = '(weighted)'
            max_model_str_len = PigOut.get_max_model_str_len()
            name_str = "{} {} {}".format(corpus_name, model, weight_str)

            output_str = "{:>" + str(max_model_str_len) + '}'
            output_str = output_str.format(name_str) + "\t"
            template_str = ''
            for metric in metrics:
                format_str = '{' + metric + ':.' + str(decimals) + 'f}'
                template_str += format_str + "\t"
            output_str += template_str.format(**rounded_results)
        print(output_str)

    @staticmethod
    def pig_fingers(pig_path):
        fingers = list()
        with open(pig_path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if line[0:2] == "//":
                    continue
                fields = line.split()
                finger = fields[-1]
                fingers.append(finger)
        return fingers

    @staticmethod
    def pig_strike_digits(pig_path):
        fingers = PigOut.pig_fingers(pig_path=pig_path)
        digits = list()
        for finger in fingers:
            tokens = finger.split('_')
            digit = tokens[0]
            digits.append(digit)
        return digits

    @staticmethod
    def nakamura_accuracy(fingering_files_dir=PIG_FINGERING_DIR, prediction_dir=PIG_PREDICTION_DIR,
                          model='fhmm3', output="text"):
        prediction_path = Path(prediction_dir)
        if not prediction_path.is_dir():
            os.makedirs(prediction_dir)

        corpus_name = PigOut.corpus_name_for_dir(fingering_files_dir)
        y_test = list()
        y_pred = list()

        for root, dirs, files in os.walk(fingering_files_dir):
            for file in sorted(files):
                if file.endswith(PIG_FILE_SUFFIX):
                    mat = re.match(PIG_FILE_RE, file)
                    piece_id = mat.group(PIG_FILE_PIECE_LOC)
                    piece_number = int(piece_id)
                    if piece_number > 30:  # First 30 are in test corpus.
                        break
                    input_path = os.path.join(root, file)
                    output_path = prediction_dir + file
                    PigOut.run_hmm(model=model, in_fin=input_path, out_fin=output_path)
                    pred_strike_digits = PigOut.pig_strike_digits(output_path)
                    y_pred.append(pred_strike_digits)
                    test_strike_digits = PigOut.pig_strike_digits(input_path)
                    y_test.append(test_strike_digits)

        labels = PIG_STRIKE_LABELS
        flat_weighted_f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        if output == 'text':
            method_str = "{:>7} {:>5}".format(corpus_name, model)
            print("{} flat weighted F1: {}".format(method_str, flat_weighted_f1))

            # sorted_labels = sorted(
                # labels,
                # key=lambda name: (name[1:], name[0])
            # )
            print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=4))

    @staticmethod
    def piece_id_for_file_name(file_name):
        mat = re.match(PIG_FILE_RE, file_name)
        if mat:
            piece_id = mat.group(PIG_FILE_PIECE_LOC)
            return piece_id

        raise Exception("Ill-formed file: {}".format(file_name))

    @staticmethod
    def pig_files_and_paths(fingering_files_dir=PIG_FINGERING_DIR, piece_id=None):
        pig_files = list()
        pig_paths = list()
        for root, dirs, files in os.walk(fingering_files_dir):
            sorted_files = sorted(files)
            for file in sorted_files:
                if file.endswith(PIG_FILE_SUFFIX):
                    if piece_id is not None:
                        mat = re.match(PIG_FILE_RE, file)
                        if mat:
                            this_piece_id = mat.group(PIG_FILE_PIECE_LOC)
                            if this_piece_id != piece_id:
                                continue
                    pig_files.append(file)
                    path = os.path.join(root, file)
                    pig_paths.append(path)
        return pig_files, pig_paths

    @staticmethod
    def my_average_m_gen(fingering_files_dir=PIG_FINGERING_DIR, prediction_input_dir=PIG_RESULT_FHMM3_DIR, weight=False):
        piece_results = dict()
        pred_files, pred_paths = PigOut.pig_files_and_paths(fingering_files_dir=prediction_input_dir)
        total_match_count = 0
        total_note_count = 0
        total_annot_count = 0
        piece_count = len(pred_paths)
        piece_match_counts = dict()
        piece_annot_counts = dict()
        piece_note_counts = dict()
        for i in range(piece_count):
            pred_path = pred_paths[i]
            pred_notes = PigIn.pig_notes(file_path=pred_path)
            note_count = len(pred_notes)
            piece_id = PigOut.piece_id_for_file_name(file_name=pred_files[i])
            piece_note_counts[piece_id] = note_count
            gt_files, gt_paths = PigOut.pig_files_and_paths(fingering_files_dir=fingering_files_dir, piece_id=piece_id)
            match_count, annot_count, m_gen = \
                PigOut.my_single_prediction_m_gen(gt_pig_paths=gt_paths, pred_pig_path=pred_path)
            complex = PigOut.single_prediction_complex_match_rates(gt_pig_paths=gt_paths, pred_pig_path=pred_path)
            complex_m_gen = complex['general']
            complex_match_count = round(complex_m_gen * annot_count)

            if match_count != complex_match_count:
                msg = "{} has conflicting match counts: {} vs. {}.\n".format(piece_id, match_count, complex_match_count)
                msg += "pred_file: {}, test_files: {}".format(pred_path, gt_paths)
                print(msg)

            total_annot_count += annot_count
            total_match_count += match_count
            total_note_count += note_count
            piece_match_counts[piece_id] = match_count
            piece_annot_counts[piece_id] = annot_count
        for piece_id in sorted(piece_annot_counts):
            piece_results[piece_id] = dict()
            piece_rate = piece_match_counts[piece_id] / piece_annot_counts[piece_id]
            contribution = 1.0 / piece_count
            if weight:
                contribution = piece_note_counts[piece_id]/total_note_count
            general_result = {
                'match_count': piece_match_counts[piece_id],
                'note_count': piece_note_counts[piece_id],
                'annot_count': piece_annot_counts[piece_id],
                'raw_rate': piece_rate,
                'contribution': contribution
            }
            piece_results[piece_id]['general'] = general_result
        if weight:
            avg_m_gen = 0
            for piece_id in sorted(piece_annot_counts):
                piece_rate = piece_match_counts[piece_id]/piece_annot_counts[piece_id]
                contribution = piece_note_counts[piece_id]/total_note_count
                avg_m_gen += piece_rate*contribution
        else:
            avg_m_gen = total_match_count/total_annot_count
            print("{}/{} = {}".format(total_match_count, total_annot_count, avg_m_gen))
        if avg_m_gen > 1.0:
            raise Exception("Bad average m_gen calculated: {}".format(avg_m_gen))
        return avg_m_gen, piece_results

    @staticmethod
    def my_average_m(fingering_files_dir=PIG_FINGERING_DIR, prediction_input_dir=PIG_RESULT_FHMM3_DIR, weight=False):
        piece_results = dict()
        piece_details = dict()
        pred_files, pred_paths = PigOut.pig_files_and_paths(fingering_files_dir=prediction_input_dir)
        total_note_count = 0
        total_annot_count = 0
        piece_count = len(pred_paths)
        piece_annot_counts = dict()
        piece_note_counts = dict()
        for i in range(piece_count):
            pred_path = pred_paths[i]
            pred_notes = PigIn.pig_notes(file_path=pred_path)
            note_count = len(pred_notes)
            piece_id = PigOut.piece_id_for_file_name(file_name=pred_files[i])
            piece_note_counts[piece_id] = note_count
            gt_files, gt_paths = PigOut.pig_files_and_paths(fingering_files_dir=fingering_files_dir, piece_id=piece_id)
            m_i = PigOut.my_single_prediction_m(gt_pig_paths=gt_paths, pred_pig_path=pred_path)
            piece_results[piece_id] = m_i
            piece_annot_counts[piece_id] = note_count * len(gt_paths)
            total_annot_count += piece_annot_counts[piece_id]
            total_note_count += note_count
        for piece_id in sorted(piece_results):
            piece_details[piece_id] = dict()
            for metric in NAKAMURA_METRICS:
                metric_val = piece_results[piece_id][metric]
                if metric == 'general':
                    contribution = 1.0 / piece_count
                    if weight:
                        contribution = piece_note_counts[piece_id]/total_note_count
                    details = {
                        'match_count': round(piece_annot_counts[piece_id] * metric_val),
                        'note_count': piece_note_counts[piece_id],
                        'annot_count': piece_annot_counts[piece_id],
                        'raw_rate': metric_val,
                        'contribution': contribution
                    }
                elif metric == 'recomb':
                    details = {}
                else:
                    contribution = 1.0 / piece_count
                    if weight:
                        contribution = piece_note_counts[piece_id]/total_note_count
                    details = {
                        'match_count': round(piece_note_counts[piece_id] * metric_val),
                        'note_count': piece_note_counts[piece_id],
                        'annot_count': piece_annot_counts[piece_id],
                        'raw_rate': metric_val,
                        'contribution': contribution
                    }
                piece_details[piece_id][metric] = details

        avg_m = {}
        for metric in NAKAMURA_METRICS:
            avg_m[metric] = 0

        for metric in NAKAMURA_METRICS:
            if metric == 'recomb':
                continue
            for piece_id in sorted(piece_details):
                piece_rate = piece_details[piece_id][metric]['raw_rate']
                contribution = piece_details[piece_id][metric]['contribution']
                avg_m[metric] += piece_rate * contribution
            if avg_m[metric] > 1.0:
                raise Exception("Bad average M_{} calculated: {}".format(metric, avg_m[metric]))
        return avg_m, piece_details

    @staticmethod
    def nakamura_metrics(fingering_files_dir=PIG_FINGERING_DIR, prediction_output_dir=None,
                         prediction_input_dir=PIG_RESULT_FHMM3_DIR,
                         model='fhmm3', weight=False, output="text"):
        if model == 'human':
            return PigOut.nakamura_human(fingering_files_dir=fingering_files_dir, output=output, weight=weight)

        if prediction_output_dir:
            prediction_output_path = Path(prediction_output_dir)
            if not prediction_output_path.is_dir():
                os.makedirs(prediction_output_dir)

        current_piece_id = None
        gt_files = dict()
        prediction_files = dict()
        note_counts = dict()
        total_note_count = 0
        for root, dirs, files in os.walk(fingering_files_dir):
            for file in sorted(files):
                if file.endswith(PIG_FILE_SUFFIX):
                    mat = re.match(PIG_FILE_RE, file)
                    if mat:
                        piece_id = mat.group(PIG_FILE_PIECE_LOC)
                        input_path = os.path.join(root, file)
                        annot_id = mat.group(PIG_FILE_ANNOT_LOC)
                        name = mat.group(PIG_FILE_NAME_LOC)
                        piece_number = int(piece_id)
                        if piece_number > 30:
                            break
                        if piece_id != current_piece_id:
                            with open(input_path, 'r') as fp:
                                note_count = len(fp.readlines()) - 1  # Ignore comment on first line.
                                note_counts[piece_id] = note_count
                                total_note_count += note_count
                            if prediction_output_dir:
                                output_path = prediction_output_dir + file
                                PigOut.run_hmm(model=model, in_fin=input_path, out_fin=output_path)
                            else:
                                output_path = prediction_input_dir + file
                            prediction_files[piece_id] = output_path
                            current_piece_id = piece_id
                        if piece_id not in gt_files:
                            gt_files[piece_id] = list()
                        gt_files[piece_id].append(input_path)

        total_annot_count = 0
        total_match_counts_by_metric = dict()
        piece_results = dict()
        metrics = NAKAMURA_METRICS
        for key_piece_id in gt_files:
            total_annot_count += note_counts[key_piece_id] * len(gt_files[key_piece_id])
            piece_results[key_piece_id] = {}
            result = PigOut.single_prediction_complex_match_rates(gt_pig_paths=gt_files[key_piece_id],
                                                                  pred_pig_path=prediction_files[key_piece_id])
            for metric in result:
                piece_annot_count = note_counts[key_piece_id] * len(gt_files[key_piece_id])
                piece_match_count = round(result[metric] * piece_annot_count)
                if metric not in total_match_counts_by_metric:
                    total_match_counts_by_metric[metric] = piece_match_count
                else:
                    total_match_counts_by_metric[metric] += piece_match_count
                piece_metric = {
                    'annot_count': piece_annot_count,
                    'match_count': piece_match_count,
                    'note_count': note_counts[key_piece_id],
                    'raw_rate': piece_match_count / piece_annot_count
                }
                piece_results[key_piece_id][metric] = piece_metric

        totals = dict()
        for metric in metrics:
            if weight:
                if metric not in totals:
                    totals[metric] = 0
                for key_piece_id in sorted(gt_files):
                    piece_match_count = piece_results[key_piece_id][metric]['match_count']
                    piece_annot_count = piece_results[key_piece_id][metric]['annot_count']
                    piece_note_count = piece_results[key_piece_id][metric]['note_count']
                    piece_rate = piece_match_count / piece_annot_count
                    contribution = piece_note_count/total_note_count
                    totals[metric] += piece_rate * contribution
            else:
                totals[metric] = total_match_counts_by_metric[metric]/total_annot_count
                # print("M_{} = {}/{} = {}".format(metric, total_match_counts_by_metric[metric],
                #                                  total_annot_count, totals[metric]))
            if totals[metric] >= 1.0:
                raise Exception("Metric {} is invalid: {}".format(metric, totals[metric]))

        if output:
            corpus_name = PigOut.corpus_name_for_dir(fingering_files_dir=fingering_files_dir)
            PigOut.output_nakamura_metrics(results=totals, corpus_name=corpus_name, model=model,
                                           weight=weight, output_type=output)
            # for piece_id in sorted(piece_results):
                # print("{}: {}".format(piece_id, piece_results[piece_id]))

        return totals, piece_results

    @staticmethod
    def corpus_name_for_dir(fingering_files_dir):
        if fingering_files_dir == PIG_SEGREGATED_FINGERING_DIR:
            corpus_name = 'pig_std'
        if fingering_files_dir == PIG_FINGERING_DIR:
            corpus_name = 'pig'
        elif fingering_files_dir == PIG_SEGREGATED_FINGERING_DIR:
            corpus_name = 'pig_seg'
        elif fingering_files_dir == PIG_SEGREGATED_STD_DIR:
            corpus_name = 'pig_sts'
        else:
            raise Exception("Unrecognized fingering files directory: {}".format(fingering_files_dir))
        return corpus_name

    @staticmethod
    def nakamura_human(fingering_files_dir=PIG_FINGERING_DIR, output="text", weight=True):
        current_piece_id = None
        piece_files = dict()
        comparable_piece_id_set = set()
        note_counts = dict()
        total_note_count = 0
        for root, dirs, files in os.walk(fingering_files_dir):
            for file in sorted(files):
                if file.endswith(PIG_FILE_SUFFIX):
                    mat = re.match(PIG_FILE_RE, file)
                    if mat:
                        piece_id = mat.group(PIG_FILE_PIECE_LOC)
                        input_path = os.path.join(root, file)
                        # annot_id = mat.group(PIG_FILE_ANNOT_LOC)
                        file_name = mat.group(PIG_FILE_NAME_LOC)
                        piece_number = int(piece_id)
                        if piece_number > 30:
                            break
                        if piece_id != current_piece_id:
                            if weight:
                                with open(input_path, 'r') as fp:
                                    note_count = len(fp.readlines()) - 1  # Ignore comment on first line.
                                    note_counts[piece_id] = note_count
                                    total_note_count += note_count
                            current_piece_id = piece_id
                        if piece_id not in piece_files:
                            piece_files[piece_id] = list()
                        piece_files[piece_id].append(input_path)
                        if len(piece_files[piece_id]) > 1:
                            comparable_piece_id_set.add(piece_id)

        piece_results = dict()
        for piece_id in piece_files:
            for prediction_file in piece_files[piece_id]:
                gt_files = list()
                for gt_file in piece_files[piece_id]:
                    if gt_file != prediction_file:
                        gt_files.append(gt_file)
                result = PigOut.single_prediction_complex_match_rates(gt_pig_paths=gt_files,
                                                                      pred_pig_path=prediction_file)
                if piece_id not in piece_results:
                    piece_results[piece_id] = []
                if result is not None:
                    piece_results[piece_id].append(result)

        piece_count = len(comparable_piece_id_set)
        totals = dict()

        for piece_id in piece_files:
            piece_file_count = len(piece_files[piece_id])
            for result in piece_results[piece_id]:
                for metric in result:
                    if metric not in totals:
                        totals[metric] = 0
                    if weight:
                        totals[metric] += (result[metric]*note_counts[piece_id])/(total_note_count*piece_file_count)
                    else:
                        totals[metric] += result[metric]/(piece_count*piece_file_count)
        if output:
            corpus_name = PigOut.corpus_name_for_dir(fingering_files_dir)
            PigOut.output_nakamura_metrics(results=totals, corpus_name=corpus_name, model="human",
                                           weight=weight, output_type=output)
        return totals, {}

    @staticmethod
    def get_pig_corpus_path(piece_id):
        file_name = piece_id + PIG_FILE_SUFFIX
        path = PIG_FINGERING_DIR + file_name
        return path

    @staticmethod
    def get_std_pig_corpus_path(piece_id):
        file_name = piece_id + PIG_FILE_SUFFIX
        path = PIG_STD_DIR + file_name
        return path

    @staticmethod
    def fingered_ordered_offset_note_to_pig_note(foon, note_id, staff, fake_velocity=True, handed_channels=True):
        note_on_s = foon['second_offset']
        note_off_s = note_on_s + foon['second_duration']
        m21_note = foon['note']
        pig_name = PigNote.m21_name_to_pig_name(m21_note.nameWithOctave)
        hsd = foon['handed_strike_digit']
        pig_fingering = PigNote.abcdf_to_pig_fingering(handed_digit=hsd)

        channel = "0"
        if handed_channels:
            if pig_fingering[0] == '-':
                channel = "1"
        else:
            if staff == "lower":
                channel = "1"
            elif staff == "upper":
                channel = "0"
            else:
                raise Exception("Specific staff must be specified.")

        on_velocity = m21_note.volume.velocity
        off_velocity = "64"
        if on_velocity is None:
            if fake_velocity:
                on_velocity = 64
            else:
                raise Exception("Velocity is not set for note {} at index {} on channel {}.".format(
                    pig_name, note_id, channel))
        on_velocity = str(on_velocity)

        pig_note = PigNote(id=note_id, on=note_on_s, off=note_off_s,
                           name=pig_name, on_vel=on_velocity, off_vel=off_velocity,
                           channel=channel, finger=pig_fingering)
        return pig_note

    @staticmethod
    def transform_fingered_ordered_offset_note_list_set(foonls, combine_staffs=False,
                                                        align_segments=False,
                                                        handed_channels=True, to_dir=None, start_over=True):
        """
        Output a set of standardized PIG records from the fingered ordered offset note list set.
        Returns a dictionary of such records indexed by (score_title, staff, annotation_index, segment_index)
        tuples, like the input foonls.

        If combine_staffs is True, all records with matching (score_title, annotation_index, segment_index)
        index elements are combined into records with (score_title, 'both', annotation_index, segment_index)
        keys.

        If align_segments is True, upper and lower staff segments with the same segment_index, are combined into
        (score_title, 'both', annotation_index, segment_index) records. Any unpaired records are returned with
        the appropriate staff setting.

        If handed_channels is True, the channel assignment for a note is made according to the hand of the
        specified fingering. This is the intended semantics of the files in the original PIG dataset per
        personal correspondence with Nakamura. Therefore, this is the default behavior. We note that this
        renders channel assignments in PIG files redundant (as they may be directly inferred by the fingerings)
        and also that this limits models built with PIG files to solving the subproblem of *segregated piano fingering.
        If handed_channels is False, channel assignments are made according to the staff (0 for upper and 1 f lower).

        If to_dir is specified, PIG files are saved in this location. They are named by joining the key fields
        with double underscores, replacing all whitespace with single underscores, and appending the
        PIG file extension, "_annotation.txt". For example,

             Sonatina_1__both__0_None_fingering.txt
        """
        all_pig_strings = dict()
        for foonl_key in foonls:
            (score_title, staff, annotation_index, segment_index) = foonl_key
            pig_notes = list()
            note_id = 0
            for foon in foonls[foonl_key]:
                pig_note = PigOut.fingered_ordered_offset_note_to_pig_note(foon=foon, note_id=note_id, staff=staff,
                                                                           handed_channels=handed_channels)
                note_id += 1
                pig_notes.append(pig_note)
            contents = PigOut.pig_notes_to_string(pig_notes=pig_notes)
            all_pig_strings[foonl_key] = contents
        if combine_staffs:
            raise Exception("combine_staffs not implemented yet.")
        elif align_segments:
            raise Exception("align_segments not implemented yet.")
        else:
            pig_strings = all_pig_strings

        if to_dir:
            PigIn.mkdir_if_missing(to_dir, make_missing=start_over)
            for file_key in pig_strings:
                file_name = "__".join(file_key)
                file_name = file_name.replace(" ", "_")
                to_file = to_dir + "/" + file_name
                f = open(to_file, "w")
                f.write(contents)
                f.close()

        return pig_strings

    def transform(self, annotation_index=0, annotation_id=None, to_file=None):
        if annotation_id is not None:
            annot = self._d_score.annotation_by_id(annotation_id)
        elif annotation_index is not None:
            annot = self._d_score.annotation_by_index(annotation_index)
        else:
            raise Exception("Annotation to transform is not specified.")
        channel = 0
        note_id = 0
        pig_notes = []
        for staff in ['upper', 'lower']:
            hsds = annot.handed_strike_digits(staff=staff)
            hsd_count = len(hsds)
            ordered_offset_notes = self._d_score.ordered_offset_notes(staff=staff)
            note_count = len(ordered_offset_notes)
            if hsd_count != note_count:
                raise Exception("Note count does not equal annotation count.")
            for i in range(len(ordered_offset_notes)):
                oon = ordered_offset_notes[i]
                m21_note: note.Note = oon['note']
                pig_name = PigNote.m21_name_to_pig_name(m21_note.nameWithOctave)
                note_on_s = oon['second_offset']
                duration_s = oon['second_duration']
                note_off_s = note_on_s + duration_s
                on_velocity = m21_note.volume.velocity
                if on_velocity is None:
                    raise Exception(
                        "Velocity is not set for note {} at index {} on channel {}.".format(pig_name, i, channel))
                off_velocity = 64
                hsd = hsds[i]
                pig_fingering = PigNote.abcdf_to_pig_fingering(handed_digit=hsd)
                note_on_s = round(note_on_s, 6)
                note_off_s = round(note_off_s, 6)
                pig_note = PigNote(id=note_id, on=note_on_s, off=note_off_s,
                                   name=pig_name, on_vel=on_velocity, off_vel=off_velocity,
                                   channel=channel, finger=pig_fingering)
                pig_notes.append(pig_note)
                note_id += 1
            channel += 1
        pig_notes.sort(key=lambda x: (x.on, x.midi_pitch))
        contents = PigOut.pig_notes_to_string(pig_notes=pig_notes)
        # print(contents)
        if to_file:
            f = open(to_file, "w")
            f.write(contents)
            f.close()
        return contents

    @staticmethod
    def pig_notes_to_string(pig_notes):
        contents = PigNote.header_line()
        note_id = 0
        for pn in pig_notes:
            pn.id = note_id
            contents += pn.to_file_line()
            note_id += 1
        return contents

    @staticmethod
    def pig_notes_to_file(pig_notes, to_file):
        contents = PigOut.pig_notes_to_string(pig_notes=pig_notes)
        f = open(to_file, "w")
        f.write(contents)
        f.close()

    @staticmethod
    def zero_all_channels(pig_path, to_file):
        pig_notes = PigIn.pig_notes(file_path=pig_path)
        for pig_note in pig_notes:
            pig_note.channel = 0
        PigOut.pig_notes_to_file(pig_notes=pig_notes, to_file=to_file)
