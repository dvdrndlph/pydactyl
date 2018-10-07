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
from music21 import *
from pydactyl.dactyler import Constant

from .DScore import DScore
from .ABCDHeader import ABCDHeader
from .ManualDSegmenter import ManualDSegmenter


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
                        d_score = DScore(music21_stream=score, segmenter=self.segmenter(),
                                         abcd_header=DCorpus.abcd_header(corpus_path=corpus_path))
                        self._d_scores.append(d_score)
                else:
                    score = corp
                    d_score = DScore(music21_stream=score, segmenter=self.segmenter(),
                                     abcd_header=DCorpus.abcd_header(corpus_path=corpus_path))
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
                                     voice_map=staff_assignments[score_index], segmenter=self.segmenter())
                else:
                    d_score = DScore(abc_handle=ah_for_id[score_id], abcd_header=abcd_header,
                                     segmenter=self.segmenter())
                self._d_scores.append(d_score)
                score_index += 1

    def pitch_range(self, staff="both"):
        overall_low = None
        overall_high = None
        for score in self._d_scores:
            low, high = score.pitch_range(staff=staff)
            if overall_low is None or low < overall_low:
                overall_low = low
            if overall_high is None or high > overall_high:
                overall_high = high
        return overall_low, overall_high

    def segmenter(self, segmenter=None):
        if segmenter:
            self._segmenter = segmenter
            for d_score in self._d_scores:
                d_score.segmenter(segmenter)
        return self._segmenter

    def __init__(self, corpus_path=None, corpus_str=None, paths=[], segmenter=None):
        self._conn = None
        self._d_scores = []
        self._segmenter = segmenter
        if corpus_path:
            self.append(corpus_path=corpus_path)
        if corpus_str:
            self.append(corpus_str=corpus_str)
        for path in paths:
            self.append(corpus_path=path)

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

