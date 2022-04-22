#!/usr/bin/env python
__author__ = 'David Randolph'
# Copyright (c) 2020 David A. Randolph.
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
#
# Validate event count in dig_on_swine output files.
import copy
import re
import sys
from music21 import abcFormat, converter, stream
from pydactyl.dcorpus.DCorpus import DCorpus, DAnnotation
from mido import MidiFile

PIG_DIR = '/Users/dave/tb2/didactyl/dd/corpora/pig/PianoFingeringDataset_v1.00/abcd/'
mf_path = PIG_DIR + '001-1.mid'
hdr_path = PIG_DIR + '001-1.abcd'
s = converter.parse(mf_path)

mid = MidiFile(mf_path)
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    msg_cnt = 0
    for msg in track:
        if msg.type == 'note_on':
            msg_cnt += 1
            # print(msg)
    print("note_on count: {}".format(msg_cnt))

corpse = DCorpus()
corpse.append(corpus_path=mf_path, header_path=hdr_path)
print("Done")
