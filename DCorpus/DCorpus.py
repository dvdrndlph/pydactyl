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
from music21 import *
from Dactyler import Constant


class DPart:
    def __init__(self, music21_stream, segmenter=None):
        self.stream = music21_stream
        self.segmenter = segmenter

    def _quantize_stream(self):
        return self.stream

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

    def get_stream(self, quantize=False):
        if quantize:
            return self._quantize_stream()
        return self.stream


class DScore:
    def __init__(self, music21_stream=None, abc_handle=None):
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

            if len(voices) == 2:
                upper_ah = headers + voices[0]  # First voice must be for right hand
                upper_stream = abcFormat.translate.abcToStreamScore(upper_ah)
                self.upper_d_part = DPart(music21_stream=upper_stream)
                lower_ah = headers + voices[1]  # Second voice must be for left hand
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


class DCorpus:
    """A corpus for the rest of us."""

    def __init__(self, corpus_path, corpus_type=Constant.CORPUS_ABC):
        self.d_scores = []
        if corpus_type == Constant.CORPUS_ABC:
            abc_file = abcFormat.ABCFile()
            abc_file.open(filename=corpus_path)
            abc_handle = abc_file.read()
            abc_file.close()
            ah_for_id = abc_handle.splitByReferenceNumber()
            for voice_id in ah_for_id:
                d_score = DScore(abc_handle=ah_for_id[voice_id])
                self.d_scores.append(d_score)
        else:
            self.corpus = converter.parse(corpus_path)
            if isinstance(self.corpus, stream.Opus):
                for score in self.corpus:
                    d_score = DScore(music21_stream=score)
                    self.d_scores.append(d_score)
            else:
                score = self.corpus
                d_score = DScore(music21_stream=score)
                self.d_scores.append(d_score)

    def get_score_count(self):
        return len(self.d_scores)

    def get_d_score_by_title(self, title):
        for d_score in self.d_scores:
            if d_score.title == title:
                return d_score
        return None

