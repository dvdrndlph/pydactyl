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
import pprint
from tatsu import parse
from abc import ABC, abstractmethod
from music21 import *
from Dactyler import Constant


class DValuation:
    def __init__(self, gold_annotation, test_annotation):
        self._gold = gold_annotation
        self._test = test_annotation

    @abstractmethod
    def measure(self):
        pass


class DHammingValuation(DValuation):
    def __init__(self, gold_annotation, test_annotation):
        super().__init__(gold_annotation, test_annotation)

    def measure(self):
        pass


class DNaturalValuation(DValuation):
    def __init__(self, gold_annotation, test_annotation):
        super().__init__(gold_annotation, test_annotation)

    def measure(self):
        pass


class DPivotValuation(DValuation):
    def __init__(self, gold_annotation, test_annotation):
        super().__init__(gold_annotation, test_annotation)

    def measure(self):
        pass


class DReEntryValuation(DValuation):
    def __init__(self, gold_annotation, test_annotation):
        super().__init__(gold_annotation, test_annotation)

    def measure(self):
        pass


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

    def as_d_part(self):
        return self._combined_d_part

    def upper_d_part(self):
        return self._upper_d_part

    def lower_d_part(self):
        return self._lower_d_part

    def stream(self):
        return self._combined_d_part.stream()

    def upper_stream(self):
        if self._upper_d_part:
            return self._upper_d_part.stream()
        return None

    def lower_stream(self):
        if self._lower_d_part:
            return self._lower_d_part.stream()
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

    def is_fully_annotated(self):
        if not self.is_annotated():
            return False
        # FIXME: We need to parse the abcDF to do this accurately.
        return True

    def fingering(self, index=0, identifier=1):
        if self._abcd_header:
            return self._abcd_header.fingering(index=index)
        return None

    def upper_fingering(self, index=0, identifier=1):
        if self._abcd_header:
            return self._abcd_header.upper_fingering(index=index)
        return None

    def lower_fingering(self, index=0, identifier=1):
        if self._abcd_header:
            return self._abcd_header.lower_fingering(index=index)
        return None

    def title(self):
        return self._title


class ABCDFAnnotation:
    GRAMMAR = """
        @@grammar::Calc
        
        sequence = upper:staff ['@' lower:staff] ;
        
        staff = '&'.{line};
        line = {score_fingering}* ;
        
        score_fingering = orn:ornamental ["/" alt_orn:ornamental]
        | pf:pedaled_fingering ['/' alt_pf:pedaled_fingering]
        | p:pedaling ['/' alt_p:pedaling]
        ;
         
        ornamental = ornaments:('(' {pedaled_fingering}+ ')') ;
        
        pedaled_fingering = soft:[soft] fingering:fingering damper:[damper] ;
        pedaling = soft:{soft}+ 'x' damper:{damper}+ ;
        
        fingering = strike:finger ['-' release:finger] ;
        finger = hand:[hand] digit:digit ;
        
        damper = '_' | '^' ;
        soft = 'p' | 'f' ;
        hand = '<' | '>' ;
        digit = '1' | '2' | '3' | '4' | '5' ;
    """

    @staticmethod
    def ast_for_abcdf(abcdf):
        ast = parse(ABCDFAnnotation.GRAMMAR, abcdf)
        # print(abcdf)
        # pprint.pprint(ast, indent=1)
        return ast

    def parse(self):
        return ABCDFAnnotation.ast_for_abcdf(self._abcdf)

    def parse_upper(self):
        upper_abcdf = self.upper_abcdf()
        return ABCDFAnnotation.ast_for_abcdf(upper_abcdf)

    def parse_lower(self):
        lower_abcdf = self.upper_abcdf()
        return ABCDFAnnotation.ast_for_abcdf(lower_abcdf)

    def pedaled_fingering_count(self, staff="both"):
        ast = self.parse()
        count = 0
        # Each staff is parsed into an array of lines. Each
        # line is an array of "score fingerings," or note
        # fingerings with all the trimmings.
        if staff == "upper" or staff == "both":
            lines = ast.upper
            for line in lines:
                for score_fingering in line:
                    count += 1
        if staff == "lower" or staff == "both":
            lines = ast.lower
            for line in lines:
                for score_fingering in line:
                    count += 1
        return count

    def segregated_strike_digits(self, staff="upper", hand=None):
        """
        :return: String of digits (1-5), assuming all fingerings are
                 are for the specified hand (">" or right for the
                 upper staff by default).

                 Returns None if any fingerings for the other hand
                 are detected.
        """
        if staff not in ("upper", "lower"):
            raise Exception("Invalid input: staff must be 'upper' or 'lower'.")

        if not hand:
            hand = ">"
            if staff == "lower":
                hand = "<"

        digits = []
        ast = self.parse()
        if staff == "upper":
            lines = ast.upper
        else:
            lines = ast.lower

        for line in lines:
            for score_fingering in line:
                strike = score_fingering.pf.fingering.strike
                current_hand = strike.hand
                digit = strike.digit
                if current_hand and current_hand != hand:
                    return None
                digits.append(digit)
        digit_str = "".join(digits)
        return digit_str

    def __init__(self, abcdf=None):
        self._authority = None
        self._authority_year = None
        self._transcriber = None
        self._transcription_date = None
        self._abcdf = abcdf
        self._abcdf_id = None
        self._comments = ''

    def authority(self, authority=None):
        if authority:
            self._authority = authority
        return self._authority

    def authority_year(self, authority_year=None):
        if authority_year:
            self._authority_year = authority_year
        return self._authority_year

    def transcriber(self, transcriber=None):
        if transcriber:
            self._transcriber = transcriber
        return self._transcriber

    def transcription_date(self, transcription_date=None):
        if transcription_date:
            self._transcription_date = transcription_date
        return self._transcription_date

    def abcdf(self, abcdf=None):
        if abcdf:
            self._abcdf = abcdf
        return self._abcdf

    def abcdf_id(self, abcdf_id=None):
        if abcdf_id:
            self._abcdf_id = abcdf_id
        return self._abcdf_id

    def comments(self, comments=None):
        if comments:
            self._comments = comments
        return self._comments.rstrip()

    def add_comment_line(self, comment):
        self._comments += comment + "\n"

    def upper_abcdf(self):
        (upper, lower) = self.abcdf().split('@')
        return upper

    def lower_abcdf(self):
        (upper, lower) = self.abcdf().split('@')
        return lower


class ABCDHeader:
    COMMENT_RE = r'^%\s*(.*)'
    TITLE_RE = r'^%\s*abcDidactyl v(\d+)'
    FINGERING_RE = r'^%\s*abcD fingering (\d+):\s*(.*)'
    TERMINAL_RE = r'^%\s*abcDidactyl END'
    AUTHORITY_RE = r'^%\s*Authority:\s*([^\(]+)\s*(\((\d+)\))?'
    TRANSCRIBER_RE = r'^%\s*Transcriber:\s*(.*)'
    TRANSCRIPTION_DATE_RE = r'^%\s*Transcription date:\s*((\d\d\d\d\-\d\d\-\d\d)\s*(\d\d:\d\d:\d\d)?)'

    @staticmethod
    def is_abcD(string):
        for line in string.splitlines():
            matt = re.search(ABCDHeader.TITLE_RE, line)
            if matt:
                return True
        return False

    def __init__(self, abcd_str):
        self._annotations = []

        annotation = ABCDFAnnotation()
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
                annotation = ABCDFAnnotation(abcdf=matt.group(2))
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

    def fingering(self, index=0, identifier=None):
        if identifier is not None:
            annotation = self.annotation_by_id(identifier)
            if annotation:
                return annotation.abcdf()
        if index >= self.annotation_count():
            return None
        return self._annotations[index].abcdf()

    def upper_fingering(self, index=0, identifier=None):
        if identifier is not None:
            annotation = self.annotation_by_id(identifier)
            if annotation:
                return annotation.upper_abcdf()
        if index >= self.annotation_count():
            return None
        return self._annotations[index].upper_abcdf()

    def lower_fingering(self, index=0, identifier=None):
        if identifier is not None:
            annotation = self.annotation_by_id(identifier)
            if annotation:
                return annotation.lower_abcdf()
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
        if ABCDHeader.is_abcD(corpus_str):
            hdr = ABCDHeader(abcd_str=corpus_str)
            return hdr
        return None

    @staticmethod
    def corpus_type(corpus_path=None, corpus_str=None):
        if corpus_path:
            corpus_str = DCorpus.file_to_string(file_path=corpus_path)
        if ABCDHeader.is_abcD(corpus_str):
            return Constant.CORPUS_ABCD
        else:
            return Constant.CORPUS_ABC
        # FIXME: Support MIDI, xml, and mxl

    def append(self, corpus_path=None, corpus_str=None, corpus_type=Constant.CORPUS_ABC):
        if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
            abc_file = abcFormat.ABCFile()
            if corpus_path:
                staff_assignments = DCorpus._score_staff_assignments(abc_file_path=corpus_path)
                abc_file.open(filename=corpus_path)
                abc_handle = abc_file.read()
                abc_file.close()
            elif corpus_str:
                staff_assignments = DCorpus._score_staff_assignments(abc_content=corpus_str)
                abc_handle = abc_file.readstr(corpus_str)
            else:
                return False

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

    def __init__(self, corpus_path=None, corpus_type=Constant.CORPUS_ABC):
        self._conn = None
        self._d_scores = []
        if corpus_path:
            self.append(corpus_path=corpus_path, corpus_type=corpus_type)

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
        # print(curs.description)

        for row in curs:
            abc_content = row[0]
            corpus_type = DCorpus.corpus_type(corpus_str=abc_content)
            self.append(corpus_str=abc_content, corpus_type=corpus_type)

        curs.close()

