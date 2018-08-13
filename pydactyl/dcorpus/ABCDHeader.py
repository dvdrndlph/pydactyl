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
from .DAnnotation import DAnnotation


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
