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
import re
import pymysql
import os
from abc import abstractmethod
from music21 import *
from Dactyler import Constant
from DCorpus.DAnnotation import DAnnotation


class DPart:
    def __init__(self, music21_stream, segmenter=None):
        self._stream = music21_stream
        self._segmenter = segmenter

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

    def orderly_note_stream(self):
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

        notes = self._stream.flat.getElementsByClass(note.Note)
        last_offset = 0
        for i in range(len(notes)):
            # print("{0}: {1}".format(str(notes[i].offset), str(notes[i].pitch)))
            start = notes[i].offset
            notes_at_offset = self._stream.flat.getElementsByOffset(offsetStart=start)
            note_index = 0
            for new_note in notes_at_offset:
                new_offset = start + note_index * short_dur.quarterLength
                if new_offset <= last_offset:
                    new_offset = last_offset + short_dur.quarterLength
                new_note.offset = new_offset
                last_offset = new_offset
                note_index += 1

        return new_note_stream

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


class DScore:
    @staticmethod
    def _voice_id(abc_voice):
        reggie = r'^V:\s*(\d+)'
        matt = re.search(reggie, abc_voice.tokens[0].src)
        if matt:
            voice_id = matt.group(1)
            return int(voice_id)
        return None

    def __init__(self, music21_stream=None, abc_handle=None, voice_map=None, abcd_header=None):
        self._abcd_header = abcd_header
        if music21_stream:
            self._combined_d_part = DPart(music21_stream=music21_stream)
            self._score = music21_stream
            meta = self._score[0]
            self._title = meta.title
        elif abc_handle:
            self._title = abc_handle.getTitle()
            music21_stream = abcFormat.translate.abcToStreamScore(abc_handle)
            self._combined_d_part = DPart(music21_stream=music21_stream)

            self._lower_d_part = None
            self._upper_d_part = None

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
                voice_map = {1: Constant.STAFF_UPPER, 2: Constant.STAFF_LOWER}
            if len(voices) >= 2:
                upper_ah = headers
                lower_ah = headers
                for voice in voices:
                    voice_id = DScore._voice_id(voice)
                    if voice_map[voice_id] == Constant.STAFF_UPPER:
                        upper_ah = upper_ah + voice
                    else:
                        lower_ah = lower_ah + voice

                upper_stream = abcFormat.translate.abcToStreamScore(upper_ah)
                self._upper_d_part = DPart(music21_stream=upper_stream)
                lower_stream = abcFormat.translate.abcToStreamScore(lower_ah)
                self._lower_d_part = DPart(music21_stream=lower_stream)

    def is_monophonic(self):
        return self._combined_d_part.is_monophonic()

    def d_part(self, staff):
        if staff == "upper":
            return self.upper_d_part()
        elif staff == "lower":
            return self.lower_d_part()
        elif staff == "both":
            return self.combined_d_part()

    def combined_d_part(self):
        return self._combined_d_part

    def upper_d_part(self):
        return self._upper_d_part

    def lower_d_part(self):
        return self._lower_d_part

    def stream(self, staff="both"):
        if staff == "upper":
            return self.upper_stream()
        elif staff == "lower":
            return self.lower_stream()
        return self._combined_d_part.stream()

    def pitch_range(self):
        return self._combined_d_part.pitch_range()

    def upper_stream(self):
        if self._upper_d_part:
            return self._upper_d_part.stream()
        return None

    def orderly_note_stream(self, staff="both"):
        if staff == "upper":
            return self.upper_orderly_note_stream()
        elif staff == "lower":
            return self.lower_orderly_note_stream()
        return self._combined_d_part.orderly_note_stream()

    def upper_orderly_note_stream(self):
        if self._upper_d_part:
            return self._upper_d_part.orderly_note_stream()
        return None

    def lower_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.stream()
        return None

    def lower_orderly_note_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.orderly_note_stream()
        return None

    def part_count(self):
        if self._upper_d_part and self._lower_d_part:
            return 2
        return 1

    def abcd_header(self):
        return self._abcd_header

    def is_annotated(self):
        if self._abcd_header:
            return True
        return False

    def note_count(self, staff="both"):
        upper = self.upper_d_part()
        lower = self.lower_d_part()
        combo = self.combined_d_part()
        note_count = 0
        if staff == 'upper' and upper:
            note_count = len(upper.orderly_note_stream())
        if staff == 'lower' and lower:
            note_count = len(lower.orderly_note_stream())
        if staff == 'both' and combo:
            note_count = len(combo.orderly_note_stream())
        return note_count

    def is_fully_annotated(self, indices=[]):
        """
        :return: True iff a strike digit is assigned to every note in the score.
        """
        if not self.is_annotated():
            return False
        upper = self.upper_d_part()
        lower = self.lower_d_part()
        combo = self.combined_d_part()
        annot_index = 0
        if upper:
            upper_note_count = len(upper.orderly_note_stream())
            lower_note_count = len(lower.orderly_note_stream())
            for annot in self._abcd_header.annotations():
                if indices and annot_index not in indices:
                    continue
                annot_index += 1
                upper_sf_count = annot.score_fingering_count(staff="upper")
                if upper_sf_count != upper_note_count:
                    return False
                for i in range(upper_sf_count):
                    strike_digit = annot.strike_digit_at_index(index=i, staff="upper")
                    if not strike_digit or strike_digit == 'x':
                        return False
                lower_sf_count = annot.score_fingering_count(staff="lower")
                if lower_sf_count != lower_note_count:
                    return False
                for i in range(lower_sf_count):
                    strike_digit = annot.strike_digit_at_index(index=i, staff="lower")
                    if not strike_digit or strike_digit == 'x':
                        return False
        else:
            note_count = len(combo.orderly_note_stream())
            for annot in self._abcd_header.annotations():
                if annotation_index is not None and annot_index != annotation_index:
                    continue
                sf_count = annot.score_fingering_count(staff="both")
                if sf_count != note_count:
                    return False
        return True

    def abcdf(self, index=0, identifier=None, staff="both"):
        if not self._abcd_header:
            return None

        if staff == "both":
            if identifier:
                return self._abcd_header.abcdf(identifier=identifier)
            else:
                return self._abcd_header.abcdf(index=index)
        elif staff == "upper":
            return self.upper_abcdf(index=index, identifier=identifier)
        elif staff == "lower":
            return self.lower_abcdf(index=index, identifier=identifier)

        return None

    def upper_abcdf(self, index=0, identifier=None):
        if self._abcd_header:
            if identifier:
                return self._abcd_header.upper_abcdf(identifier=identifier)
            else:
                return self._abcd_header.upper_abcdf(index=index)
        return None

    def lower_abcdf(self, index=0, identifier=None):
        if self._abcd_header:
            if identifier:
                return self._abcd_header.lower_abcdf(identifier=identifier)
            else:
                return self._abcd_header.lower_abcdf(index=index)
        return None

    def title(self):
        return self._title


class ABCDHeader:
    COMMENT_RE = r'^%\s*(.*)'
    TITLE_RE = r'^%\s*abcDidactyl v(\d+)'
    FINGERING_RE = r'^%\s*abcD fingering (\d+):\s*(.*)'
    TERMINAL_RE = r'^%\s*abcDidactyl END'
    AUTHORITY_RE = r'^%\s*Authority:\s*([^\(]+)\s*(\((\d+)\))?'
    TRANSCRIBER_RE = r'^%\s*Transcriber:\s*(.*)'
    TRANSCRIPTION_DATE_RE = r'^%\s*Transcription date:\s*((\d\d\d\d\-\d\d\-\d\d)\s*(\d\d:\d\d:\d\d)?)'

    @staticmethod
    def is_abcd(string):
        for line in string.splitlines():
            matt = re.search(ABCDHeader.TITLE_RE, line)
            if matt:
                return True
        return False

    def __init__(self, abcd_str):
        self._annotations = []

        annotation = DAnnotation()
        in_header = False
        for line in abcd_str.splitlines():
            matt = re.search(ABCDHeader.TITLE_RE, line)
            if matt:
                in_header = True
                self._version = matt.group(1)
                continue
            if not in_header:
                continue
            matt = re.search(ABCDHeader.TERMINAL_RE, line)
            if matt:
                break
            matt = re.search(ABCDHeader.FINGERING_RE, line)
            if matt:
                annotation = DAnnotation(abcdf=matt.group(2))
                annotation.abcdf_id(matt.group(1).rstrip())
                self._annotations.append(annotation)
                continue
            matt = re.search(ABCDHeader.AUTHORITY_RE, line)
            if matt:
                annotation.authority(matt.group(1).rstrip())
                if matt.group(2):
                    annotation.authority_year(matt.group(3))
                continue
            matt = re.search(ABCDHeader.TRANSCRIBER_RE, line)
            if matt:
                annotation.transcriber(matt.group(1).rstrip())
                continue
            matt = re.search(ABCDHeader.TRANSCRIPTION_DATE_RE, line)
            if matt:
                annotation.transcription_date(matt.group(1).rstrip())
                continue
            matt = re.search(ABCDHeader.COMMENT_RE, line)
            if matt:
                annotation.add_comment_line(matt.group(1))

    def version(self):
        return self._version

    def annotation_count(self):
        return len(self._annotations)

    def annotations(self):
        return self._annotations

    def annotation_by_id(self, identifier=1):
        for annotation in self._annotations:
            abcdf_id = annotation.abcdf_id()
            if str(abcdf_id) == str(identifier):
                return annotation
        return None

    def annotation(self, index=0, identifier=None):
        if identifier is not None:
            return self.annotation_by_id(identifier)
        if index >= self.annotation_count():
            return None
        return self._annotations[index]

    def abcdf(self, index=0, identifier=None, staff="both"):
        if staff == "both":
            if identifier is not None:
                annotation = self.annotation_by_id(identifier)
                if annotation:
                    return annotation.abcdf()
                else:
                    return None
            if index >= self.annotation_count():
                return None
            return self._annotations[index].abcdf()
        elif staff == "upper":
            return self.upper_abcdf(index=index, identifier=identifier)
        elif staff == "lower":
            return self.lower_abcdf(index=index, identifier=identifier)

        return None

    def upper_abcdf(self, index=0, identifier=None):
        if identifier is not None:
            annotation = self.annotation_by_id(identifier)
            if annotation:
                return annotation.upper_abcdf()
            else:
                return None
        if index >= self.annotation_count():
            return None
        return self._annotations[index].upper_abcdf()

    def lower_abcdf(self, index=0, identifier=None):
        if identifier is not None:
            annotation = self.annotation_by_id(identifier)
            if annotation:
                return annotation.lower_abcdf()
            else:
                return None
        if index >= self.annotation_count():
            return None
        return self._annotations[index].lower_abcdf()


class DCorpus:
    """A corpus for the rest of us."""

    @staticmethod
    def file_to_string(file_path):
        string = ''
        file = open(file_path, "r")
        for line in file:
            string += line
        file.close()
        return string

    @staticmethod
    def _score_staff_assignments(abc_file_path=None, abc_content=None):
        """Return an array of hashes mapping voices to their associated
           staves. There should be one hash for each tune in the abc file.
        """
        if abc_file_path:
            abc_content = DCorpus.file_to_string(abc_file_path)

        map_for_tune = []
        reggie = r'^%%score\s*{\s*\(([\d\s]+)\)\s*\|\s*\(([\d\s]+)\)\s*}'
        for line in abc_content.splitlines():
            matt = re.search(reggie, line)
            if matt:
                upper_voice_str = matt.group(1)
                lower_voice_str = matt.group(2)
                upper_voices = upper_voice_str.split()
                lower_voices = lower_voice_str.split()
                staff_for_voice = {}
                for voice in upper_voices:
                    staff_for_voice[int(voice)] = Constant.STAFF_UPPER
                for voice in lower_voices:
                    staff_for_voice[int(voice)] = Constant.STAFF_LOWER

                map_for_tune.append(staff_for_voice)

        return map_for_tune

    @staticmethod
    def abcd_header(corpus_path=None, corpus_str=None):
        if corpus_path:
            corpus_str = DCorpus.file_to_string(file_path=corpus_path)
        if ABCDHeader.is_abcd(corpus_str):
            hdr = ABCDHeader(abcd_str=corpus_str)
            return hdr
        return None

    @staticmethod
    def corpus_type(corpus_path=None, corpus_str=None):
        if corpus_path:
            corpus_str = DCorpus.file_to_string(file_path=corpus_path)
        if ABCDHeader.is_abcd(corpus_str):
            return Constant.CORPUS_ABCD
        else:
            return Constant.CORPUS_ABC
        # FIXME: Support MIDI, xml, and mxl

    def append_dir(self, corpus_dir):
        for file_name in os.listdir(corpus_dir):
            file_path = corpus_dir + "/" + file_name
            self.append(corpus_path=file_path)

    def append(self, corpus_path=None, corpus_str=None):
        if corpus_path:
            corpus_type = DCorpus.corpus_type(corpus_path=corpus_path)
            if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
                abc_file = abcFormat.ABCFile()
                staff_assignments = DCorpus._score_staff_assignments(abc_file_path=corpus_path)
                abc_file.open(filename=corpus_path)
                abc_handle = abc_file.read()
                abc_file.close()
            else:
                corp = converter.parse(corpus_path)
                if isinstance(corpus, stream.Opus):
                    for score in corp:
                        d_score = DScore(music21_stream=score)
                        self._d_scores.append(d_score)
                else:
                    score = corp
                    d_score = DScore(music21_stream=score)
                    self._d_scores.append(d_score)
        elif corpus_str:
            corpus_type = DCorpus.corpus_type(corpus_str=corpus_str)
            if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
                abc_file = abcFormat.ABCFile()
                staff_assignments = DCorpus._score_staff_assignments(abc_content=corpus_str)
                abc_handle = abc_file.readstr(corpus_str)
            else:
                raise Exception("Unsupported corpus type.")
        else:
            return False

        if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
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

            abcd_header = DCorpus.abcd_header(corpus_path=corpus_path, corpus_str=corpus_str)

            score_index = 0
            for score_id in ah_for_id:
                if len(staff_assignments) > 0:
                    d_score = DScore(abc_handle=ah_for_id[score_id], abcd_header=abcd_header,
                                     voice_map=staff_assignments[score_index])
                else:
                    d_score = DScore(abc_handle=ah_for_id[score_id], abcd_header=abcd_header)
                self._d_scores.append(d_score)
                score_index += 1

    def pitch_range(self):
        overall_low = None
        overall_high = None
        for score in self._d_scores:
            low, high = score.pitch_range()
            if overall_low is None or low < overall_low:
                overall_low = low
            if overall_high is None or high > overall_high:
                overall_high = high
        return overall_low, overall_high

    def __init__(self, corpus_path=None, corpus_str=None):
        self._conn = None
        self._d_scores = []
        if corpus_path:
            self.append(corpus_path=corpus_path)
        if corpus_str:
            self.append(corpus_str=corpus_str)


    def __del__(self):
        if self._conn:
            self._conn.close()

    def score_count(self):
        return len(self._d_scores)

    def d_score_by_title(self, title):
        for d_score in self._d_scores:
            if d_score.title() == title:
                return d_score
        return None

    def d_score_by_index(self, index):
        if index < len(self._d_scores):
            return self._d_scores[index]
        return None

    def titles(self):
        titles = []
        for d_score in self._d_scores:
            titles.append(d_score.title)
        return titles

    def d_score_list(self):
        return self._d_scores

    def db_connect(self, host='127.0.0.1', port=3306, user='didactyl', passwd='', db='diii2'):
        self._conn = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db)
        return self._conn

    def append_from_db(self, host='127.0.0.1', port=3306, user='didactyl', passwd='', db='diii2',
                       query=None, client_id=None, selection_id=None):
        if not query and (not client_id or not selection_id):
            raise Exception("Query not specified.")

        if not self._conn:
            self.db_connect(host=host, port=port, user=user, passwd=passwd, db=db)

        curs = self._conn.cursor()

        if not query:
            query = """
                select abcD
                  from annotation
                 where clientId = '{0}'
                   and selectionId = '{1}'""".format(client_id, selection_id)
        curs.execute(query)

        for row in curs:
            abc_content = row[0]
            self.append(corpus_str=abc_content)

        curs.close()

