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
import music21
from Dactyler import Constant


class DPart:
    def __init__(self, music21_score):
        self.score = music21_score


class DScore:
    def __init__(self, music21_score=None, abc_handle=None):
        if music21_score:
            self.score = music21_score
            meta = self.score[0]
            self.title = meta.title
        elif abc_handle:
            self.score = abc_handle.tokenProcess()
            self.title = abc_handle.getTitle()

            self.left_hand_d_part = None
            self.right_hand_d_part = None

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

            if len(voices) > 2:
                raise Exception("Too many voices")
            elif len(voices) == 2:
                complete_rh_ah = headers + voices[0]  # First voice must be for right hand
                # print([t.src for t in complete_rh_ah.tokens])
                right_hand_part = complete_rh_ah.tokenProcess()
                self.right_hand_d_part = DPart(music21_score=right_hand_part)
                complete_lh_ah = headers + voices[1]  # Second voice must be for left hand
                # print([t.src for t in complete_lh_ah.tokens])
                left_hand_part = complete_lh_ah.tokenProcess()
                self.left_hand_d_part = DPart(music21_score=left_hand_part)

    def get_voice_count(self):
        if self.right_hand_d_part and self.left_hand_d_part:
            return 2
        return 1


class DCorpus:
    """A corpus for the rest of us."""

    def __init__(self, corpus_path, corpus_type=Constant.CORPUS_ABC):
        self.d_scores = []
        if corpus_type == Constant.CORPUS_ABC:
            abc_file = music21.abcFormat.ABCFile()
            abc_file.open(filename=corpus_path)
            abc_handle = abc_file.read()
            abc_file.close()
            ah_for_id = abc_handle.splitByReferenceNumber()
            for voice_id in ah_for_id:
                d_score = DScore(abc_handle=ah_for_id[voice_id])
                self.d_scores.append(d_score)

        else:
            self.corpus = music21.converter.parse(corpus_path)
            if isinstance(self.corpus, music21.stream.Opus):
                for score in self.corpus:
                    d_score = DScore(music21_score=score)
                    self.d_scores.append(d_score)
            else:
                score = self.corpus
                d_score = DScore(music21_score=score)
                self.d_scores.append(d_score)

    def get_score_count(self):
        return len(self.d_scores)

    def get_d_score_by_title(self, title):
        for d_score in self.d_scores:
            if d_score.title == title:
                return d_score
        return None

