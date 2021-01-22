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
from music21 import abcFormat, converter, stream
from pydactyl.dactyler import Constant

from pydactyl.abc2xml import abc2xml
from pydactyl.xml2abc import xml2abc

from .DScore import DScore
from .DAnnotation import DAnnotation
from .ABCDHeader import ABCDHeader


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

    # @staticmethod
    # def abc2xml(file_path=None, abc_content=None):
    #     if file_path:
    #         abc_content = DCorpus.file_to_string(file_path)
    #     global abc_header, abc_voice, abc_scoredef, abc_percmap # keep computed grammars
    #     mxm = abc2xml.MusicXml()
    #     abc_header, abc_voice, abc_scoredef, abc_percmap = abc2xml.abc_grammar()   
    #     score = mxm.parse(abc_content)
    #     xml_str = abc2xml.fixDoctype(score)
    #     return xml_str

    @staticmethod
    def abc2xml(file_path=None, abc_content=None):
        if file_path:
            abc_content = DCorpus.file_to_string(file_path)
        xml_str = abc2xml.getXml(abc_string=abc_content)
        return xml_str

    @staticmethod
    def abc2xmlScores(file_path=None, abc_content=None, skip=None, max=None):
        if file_path:
            abc_content = DCorpus.file_to_string(file_path)
        xml_strings = abc2xml.getXmlScores(abc_string=abc_content, skip=skip, max=max)
        return xml_strings

    @staticmethod
    def xml2abc(file_path=None, xml_content=None):
        if file_path:
            xml_content = DCorpus.file_to_string(file_path)
        abc_str = xml2abc.getAbc(xml_string=xml_content)
        return abc_str

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
    def abcd_header(path=None, string=None):
        if path:
            string = DCorpus.file_to_string(file_path=path)
        if ABCDHeader.is_abcd(string):
            hdr = ABCDHeader(abcd_str=string)
            return hdr
        return None

    @staticmethod
    def abcd_header_str(path=None, string=None):
        hdr = DCorpus.abcd_header(path=path, string=string)
        hdr_str = hdr.__str__()
        return hdr_str

    @staticmethod
    def abc_body_str(path=None, string=None):
        if path:
            string = DCorpus.file_to_string(file_path=path)
        body_str = ''
        past_hdr = False
        if not ABCDHeader.is_abcd(string):
            past_hdr = True
        for line in iter(string.splitlines()):
            if past_hdr:
                body_str += line + "\n"
            elif line == ABCDHeader.TERMINAL_STR:
                past_hdr = True
        return body_str

    @staticmethod
    def corpus_type(corpus_path=None, corpus_str=None):
        if corpus_path:
            corpus_str = DCorpus.file_to_string(file_path=corpus_path)
        if ABCDHeader.is_abcd(corpus_str):
            return Constant.CORPUS_ABCD
        if corpus_str[0] == '<':
            return Constant.CORPUS_MUSIC_XML;
        return Constant.CORPUS_ABC
        # FIXME: Support MIDI, xml, and mxl

    def append_dir(self, corpus_dir):
        for file_name in os.listdir(corpus_dir):
            file_path = corpus_dir + "/" + file_name
            self.append(corpus_path=file_path)

    def append(self, corpus_path=None, corpus_str=None, header_path=None, header_str=None, as_xml=True):
        if corpus_path:
            corpus_type = DCorpus.corpus_type(corpus_path=corpus_path)
            if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
                corpus_str = DCorpus.file_to_string(corpus_path)

        if header_path:
            header_str = DCorpus.file_to_string(header_path)

        abcd_header = None
        abc_body = ''

        if corpus_str:
            corpus_type = DCorpus.corpus_type(corpus_str=corpus_str)
            if corpus_type == Constant.CORPUS_ABCD and not header_str:
                header_str = corpus_str
            if header_str:
                abcd_header = DCorpus.abcd_header(string=header_str)

            if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
                abc_body = DCorpus.abc_body_str(string=corpus_str)

            if as_xml:
                corpus_str = DCorpus.abc2xml(abc_content=corpus_str)
                corpus_type = DCorpus.corpus_type(corpus_str=corpus_str)

            if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
                # NOTE: We only do this if we are not using the XML transform.
                # THIS IS NOT RECOMMENDED.
                # The abc conversion does not manage the grouping of voices into
                # the appropriate part (staff), so we hack around this shortcoming.
                self._abc_strings.append(corpus_str)
                abc_file = abcFormat.ABCFile(abcVersion=(2,1,0))
                staff_assignments = DCorpus._score_staff_assignments(abc_content=corpus_str)
                abc_handle = abc_file.readstr(corpus_str)
            else:
                # THIS IS WHERE WE SHOULD BE.
                corp = converter.parse(corpus_str)
                if isinstance(corp, stream.Opus):
                    for score in corp:
                        d_score = DScore(music21_stream=score, segmenter=self.segmenter())
                        self._d_scores.append(d_score)
                else:
                    score = corp
                    d_score = DScore(music21_stream=score, segmenter=self.segmenter(),
                                     abcd_header=abcd_header, abc_body=abc_body)
                    self._d_scores.append(d_score)
        else:
            return False

        if corpus_type in [Constant.CORPUS_ABC, Constant.CORPUS_ABCD]:
            # WARNING: abc parsing is NOT recommended
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
                    d_score = DScore(abc_handle=ah_for_id[score_id], abcd_header=abcd_header, abc_body=abc_body,
                                     voice_map=staff_assignments[score_index], segmenter=self.segmenter())
                else:
                    d_score = DScore(abc_handle=ah_for_id[score_id], abcd_header=abcd_header, abc_body=abc_body,
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

    def __init__(self, corpus_path=None, corpus_str=None, paths=[], segmenter=None, as_xml=True):
        self._conn = None
        self._abc_strings = []
        self._d_scores = []
        self._segmenter = segmenter
        if corpus_path:
            self.append(corpus_path=corpus_path, as_xml=as_xml)
        if corpus_str:
            self.append(corpus_str=corpus_str, as_xml=as_xml)
        for path in paths:
            self.append(corpus_path=path, as_xml=as_xml)

    def __del__(self):
        if self._conn:
            self._conn.close()

    def score_count(self):
        return len(self._d_scores)

    def abc_string_by_index(self, index):
        if index < len(self._abc_strings):
            return self._abc_strings[index]
        return None

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
                       query=None, client_id=None, selection_id=None, as_xml=False):
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
            self.append(corpus_str=abc_content, as_xml=as_xml)

        curs.close()

    def assemble_and_append_from_db(self, host='127.0.0.1', port=3306, user='didactyl', passwd='', db='didactyl2',
                                    piece_query=None, fingering_query=None, as_xml=True):
        if not piece_query:
            raise Exception("Piece query with piece_id and abc_str columns not specified.")
        piece_conn = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db)
        piece_curs = piece_conn.cursor()
        piece_curs.execute(piece_query)

        for row in piece_curs:
            piece_id = row[0]
            abc_str = row[1]
            finger_conn = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=db,
                                          cursorclass=pymysql.cursors.DictCursor)
            finger_curs = finger_conn.cursor()
            finger_curs.execute(fingering_query.format(piece_id))
            header_str = ''
            header = ABCDHeader()
            abcdf_id = 1
            for f in finger_curs:
                abcdf = f['fingering']
                if not re.match('[<>]', abcdf):
                    abcdf = '>' + abcdf
                if not re.match('@', abcdf):
                    abcdf += '@'
                comment = "Weight: {}".format(f['weight'])
                annot = DAnnotation(abcdf=abcdf, authority=f['authority'], transcriber=f['transcriber'],
                                    abcdf_id=abcdf_id, comments=comment)
                header.append_annotation(annot)
                abcdf_id += 1
            header_str = header.__str__()
            self.append(corpus_str=abc_str, header_str=header_str, as_xml=as_xml)

