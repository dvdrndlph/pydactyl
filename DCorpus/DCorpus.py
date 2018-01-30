__author__ = 'David Randolph'
# Copyright (c) 2014 David A. Randolph.
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
import sys
import re
from music21 import *
from Dactyler import Constant


class DPart:
    def __init__(self, music21_stream, segmenter=None):
        self.stream = music21_stream
        self.segmenter = segmenter

    def get_orderly_note_stream(self):
        """Return part as stream of notes with no notes starting at the same
           offset. Chords turned into a sequence of notes with starting points
           separated by the shortest duration (a 2048th note) ordered from
           low to high. The lowest individual note at a given offset will remain
           in place. All other notes at a given offset will be nudged to the right.
           The goal here is to provide an orderly sequence of notes that can be
           processed by Dactylers that only support monophonic streams. They can
           ignore the stacking of notes and at least take a stab at more complex
           scores. We also want to approximate note durations in caase this information
           is useful for some models.
        """
        chords = self.stream.flat.getElementsByClass(chord.Chord)
        new_note_stream = stream.Score()
        for ch in chords:
            chord_offset = ch.offset
            chord_dur_length = ch.duration.quarterLength
            short_dur = duration.Duration()
            short_dur.type('2048th')
            note_index = 0
            for pitch_name in ch.pitchNames:
                new_note = note.Note(pitchName=pitch_name)
                new_note.offset = chord_offset + note_index * short_dur.quarterLength
                new_note_ql = chord_dur_length - note_index * short_dur.quarterLength
                new_note_dur = duration.Duration
                new_note_dur.quarterLength = new_note_ql
                new_note.duration = new_note_dur
                new_note_stream.append(new_note)
                note_index += 1

        notes = self.stream.flat.getElementsByClass(note.Note)
        for old_note in notes:
            new_note_stream.append(old_note)
        new_note_stream.show('text')

        return new_note_stream

        # orderly_notes = stream.Part()
        # new_notes = self.stream.flat.getElementsByClass(note.Note)
        # for i in range(len(new_notes)):
            # if i == len(notes) - 1:

    def is_monophonic(self):
        if self.stream.flat.getElementsByClass(chord.Chord):
            return False
        notes = self.stream.flat.getElementsByClass(note.Note)
        for i in range(len(notes)):
            # print("{0}: {1}".format(str(notes[i].offset), str(notes[i].pitch)))
            start = notes[i].offset
            end = start + notes[i].duration.quarterLength
            notes_in_range = self.stream.flat.getElementsByOffset(
                offsetStart=start, offsetEnd=end,
                includeEndBoundary=False,
                mustBeginInSpan=False,
                includeElementsThatEndAtStart=False,
                classList=[note.Note]
            )
            if len(notes_in_range) > 1:
                return False
        return True

    def get_stream(self, orderly=False):
        if orderly:
            return self.get_orderly_note_stream()
        return self.stream


class DScore:
    @staticmethod
    def _get_voice_id(abc_voice):
        reggie = r'^V:\s*(\d+)'
        matt = re.search(reggie, abc_voice.tokens[0].src)
        if matt:
            voice_id = matt.group(1)
            return voice_id
        return None

    def __init__(self, music21_stream=None, abc_handle=None, voice_map=None):
        if music21_stream:
            self.combined_d_part = DPart(music21_stream=music21_stream)
            self.score = music21_stream
            meta = self.score[0]
            self.title = meta.title
        elif abc_handle:
            self.title = abc_handle.getTitle()
            music21_stream = abcFormat.translate.abcToStreamScore(abc_handle)
            self.combined_d_part = DPart(music21_stream=music21_stream)

            self.lower_d_part = None
            self.upper_d_part = None

            ah_array = abc_handle.splitByVoice()
            voices = []
            headers = None
            for ah in ah_array:
                if ah.hasNotes():
                    voices.append(ah)
                else:
                    if not headers:
                        headers = ah
                    else:
                        headers = headers + ah

            if len(voices) > 2 and not voice_map:
                raise Exception("Mapping of voices to staves is required.")
            if not voice_map:
                voice_map = { 1: Constant.STAFF_UPPER, 2: Constant.STAFF_LOWER}
            if len(voices) >= 2:
                upper_ah = headers
                lower_ah = headers
                for voice in voices:
                    voice_id = DScore._get_voice_id(voice)
                    if voice_map[voice_id] == Constant.STAFF_UPPER:
                        upper_ah = upper_ah + voice
                    else:
                        lower_ah = lower_ah + voice

                upper_stream = abcFormat.translate.abcToStreamScore(upper_ah)
                self.upper_d_part = DPart(music21_stream=upper_stream)
                lower_stream = abcFormat.translate.abcToStreamScore(lower_ah)
                self.lower_d_part = DPart(music21_stream=lower_stream)

    def is_monophonic(self):
        return self.combined_d_part.is_monophonic()

    def get(self):
        return self.combined_d_part

    def get_upper(self):
        return self.upper_d_part

    def get_lower(self):
        return self.upper_d_part

    def get_stream(self):
        return self.combined_d_part.get_stream()

    def get_upper_stream(self):
        if self.upper_d_part:
            return self.upper_d_part.get_stream()
        return None

    def get_lower_stream(self):
        if self.lower_d_part:
            return self.lower_d_part.get_stream()
        return None

    def get_part_count(self):
        if self.upper_d_part and self.lower_d_part:
            return 2
        return 1

    def get_title(self):
        return self.title


class DCorpus:
    """A corpus for the rest of us."""

    @staticmethod
    def _get_score_staff_assignments(abc_file_path):
        """Return an array of hashes mapping voices to their associated
           staves. There should be one hash for each tune in the abc file.
        """
        map_for_tune = []
        file = open(abc_file_path, "r")
        reggie = r'^%%score\s*{\s*\(([\d\s]+)\)\s*\|\s*\(([\d\s]+)\)\s*}'
        for line in file:
            matt = re.search(reggie, line)
            if matt:
                upper_voice_str = matt.group(1)
                lower_voice_str = matt.group(2)
                upper_voices = upper_voice_str.split()
                lower_voices = lower_voice_str.split()
                staff_for_voice = {}
                for voice in upper_voices:
                    staff_for_voice[voice] = Constant.STAFF_UPPER
                for voice in lower_voices:
                    staff_for_voice[voice] = Constant.STAFF_LOWER

                map_for_tune.append(staff_for_voice)

        file.close()
        return map_for_tune

    def append(self, corpus_path, corpus_type=Constant.CORPUS_ABC):
        if corpus_type == Constant.CORPUS_ABC:
            staff_assignments = DCorpus._get_score_staff_assignments(corpus_path)
            abc_file = abcFormat.ABCFile()
            abc_file.open(filename=corpus_path)
            abc_handle = abc_file.read()
            abc_file.close()
            ah_for_id = abc_handle.splitByReferenceNumber()
            if len(staff_assignments) > 0 and len(staff_assignments) != len(ah_for_id):
                # We must know how to map all voices to a staff. Either all scores (tunes)
                # in corpus must have two or fewer voices, or we need a map. For simplicity,
                # we make this an all-or-nothing proposition. If any score in the corpus
                # needs a map, they all must provide one. For two voices, this would
                # look like this:
                #
                #    %%score { ( 1 ) | ( 2 ) }
                raise Exception("All abc scores in corpus must have %%score staff assignments or none should.")

            score_index = 0
            for score_id in ah_for_id:
                if len(staff_assignments) > 0:
                    d_score = DScore(abc_handle=ah_for_id[score_id], voice_map=staff_assignments[score_index])
                else:
                    d_score = DScore(abc_handle=ah_for_id[score_id])
                self.d_scores.append(d_score)
                score_index += 1
        else:
            corp = converter.parse(corpus_path)
            if isinstance(corpus, stream.Opus):
                for score in corp:
                    d_score = DScore(music21_stream=score)
                    self.d_scores.append(d_score)
            else:
                score = corp
                d_score = DScore(music21_stream=score)
                self.d_scores.append(d_score)

    def __init__(self, corpus_path=None, corpus_type=Constant.CORPUS_ABC):
        self.d_scores = []
        if corpus_path:
            self.append(corpus_path=corpus_path, corpus_type=corpus_type)

    def get_score_count(self):
        return len(self.d_scores)

    def get_d_score_by_title(self, title):
        for d_score in self.d_scores:
            if d_score.title == title:
                return d_score
        return None

    def get_d_score_by_index(self, index):
        if index < len(self.d_scores):
            return self.d_scores[index]
        return None

    def get_titles(self):
        titles = []
        for d_score in self.d_scores:
            titles.append(d_score.title)
        return titles

